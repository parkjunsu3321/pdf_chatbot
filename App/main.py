import sys
import threading
import queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

# ── App 모듈 경로 등록 (개발/빌드/exe 공통) ───────────────────────────────────────
def _get_app_dir() -> Path:
    """개발 환경과 PyInstaller exe 환경 모두 지원"""
    if getattr(sys, "frozen", False):
        # exe로 실행 중: exe 파일이 있는 폴더 기준
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_DIR = _get_app_dir()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dotenv import load_dotenv
load_dotenv(dotenv_path=APP_DIR / ".env")

from loaders.loader import load_pdf, split_documents
from utils.vectorstore import create_vectorstore
from chains.qa_chain import create_qa_chain

DATA_DIR = APP_DIR / "data"
TABLE_DIR = str(DATA_DIR / "extracted_tables")
IMG_DIR   = str(DATA_DIR / "extracted_images")

# ── 폰트 (Windows 한글) ──────────────────────────────────────────────────────────
F_NORMAL = ("맑은 고딕", 11)
F_BOLD   = ("맑은 고딕", 11, "bold")
F_SMALL  = ("맑은 고딕", 10)
F_TITLE  = ("맑은 고딕", 13, "bold")


class PDFChatbotApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PDF Chatbot")
        self.root.geometry("900x660")
        self.root.minsize(700, 500)
        self.root.configure(bg="#f0f0f0")

        self.qa_chain: object | None = None
        self._q: queue.Queue = queue.Queue()

        self._build_ui()
        self._poll()

    # ── UI 구성 ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self._build_topbar()    # row 0
        self._build_chat()      # row 1
        self._build_inputbar()  # row 2, 3, 4

    def _build_topbar(self):
        bar = tk.Frame(self.root, bg="#2c2c2c", height=52)
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(1, weight=1)
        bar.grid_propagate(False)

        tk.Label(bar, text="📄 PDF Chatbot", font=F_TITLE,
                 bg="#2c2c2c", fg="white").grid(row=0, column=0, padx=16, pady=12)

        self.status_var = tk.StringVar(value="PDF 파일을 업로드해주세요.")
        tk.Label(bar, textvariable=self.status_var, font=F_SMALL,
                 bg="#2c2c2c", fg="#aaaaaa").grid(row=0, column=1, sticky="w", padx=8)

        self.upload_btn = tk.Button(
            bar, text="PDF 업로드", font=F_NORMAL,
            bg="#4a90d9", fg="white", relief=tk.FLAT,
            activebackground="#357abd", activeforeground="white",
            padx=14, pady=6, cursor="hand2",
            command=self._on_upload,
        )
        self.upload_btn.grid(row=0, column=2, padx=12)

        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=140)
        self.progress.grid(row=0, column=3, padx=(0, 16))

    def _build_chat(self):
        outer = tk.Frame(self.root, bg="#f0f0f0")
        outer.grid(row=1, column=0, sticky="nsew", padx=14, pady=(10, 0))
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        self.chat = scrolledtext.ScrolledText(
            outer, wrap=tk.WORD, state=tk.DISABLED,
            font=F_NORMAL, bg="white", relief=tk.FLAT,
            padx=14, pady=12, spacing3=4,
        )
        self.chat.grid(row=0, column=0, sticky="nsew")

        self.chat.tag_config("user_lbl",  foreground="#1a73e8", font=F_BOLD)
        self.chat.tag_config("user_body", foreground="#1a1a1a", font=F_NORMAL)
        self.chat.tag_config("bot_lbl",   foreground="#188038", font=F_BOLD)
        self.chat.tag_config("bot_body",  foreground="#1a1a1a", font=F_NORMAL)
        self.chat.tag_config("system",    foreground="#999999", font=F_SMALL,
                             justify="center")

    def _build_inputbar(self):
        ttk.Separator(self.root, orient="horizontal").grid(
            row=2, column=0, sticky="ew", pady=(8, 0))

        # 생각 중 표시
        self.thinking_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.thinking_var, font=F_SMALL,
                 fg="#888888", bg="#f0f0f0").grid(
            row=3, column=0, sticky="w", padx=16, pady=(4, 0))

        # 입력창
        bar = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        bar.grid(row=4, column=0, sticky="ew", padx=14)
        bar.columnconfigure(0, weight=1)

        self.entry_var = tk.StringVar()
        self.entry = ttk.Entry(bar, textvariable=self.entry_var, font=F_NORMAL)
        self.entry.grid(row=0, column=0, sticky="ew", ipady=6, padx=(0, 8))
        self.entry.bind("<Return>", lambda _: self._on_send())
        self.entry.config(state=tk.DISABLED)

        self.send_btn = tk.Button(
            bar, text="전송", font=F_BOLD,
            bg="#188038", fg="white", relief=tk.FLAT,
            activebackground="#146c2e", activeforeground="white",
            padx=18, pady=6, cursor="hand2", state=tk.DISABLED,
            command=self._on_send,
        )
        self.send_btn.grid(row=0, column=1)

    # ── 이벤트 핸들러 ────────────────────────────────────────────────────────────

    def _on_upload(self):
        path = filedialog.askopenfilename(
            title="PDF 선택", filetypes=[("PDF 파일", "*.pdf")]
        )
        if not path:
            return

        self.upload_btn.config(state=tk.DISABLED)
        self.entry.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)
        self.status_var.set(f"처리 중: {Path(path).name} …")
        self.progress.start(10)

        threading.Thread(target=self._load_pdf_thread,
                         args=(path,), daemon=True).start()

    def _on_send(self):
        q = self.entry_var.get().strip()
        if not q or self.qa_chain is None:
            return

        self.entry_var.set("")
        self.entry.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)
        self.thinking_var.set("⏳  답변 생성 중 …")
        self._append_msg("user", q)

        threading.Thread(target=self._answer_thread,
                         args=(q,), daemon=True).start()

    # ── 백그라운드 스레드 ────────────────────────────────────────────────────────

    def _load_pdf_thread(self, pdf_path: str):
        try:
            docs, table_registry = load_pdf(
                pdf_path, table_dir=TABLE_DIR, img_dir=IMG_DIR
            )
            chunks      = split_documents(docs)
            vectorstore = create_vectorstore(chunks)
            qa_chain    = create_qa_chain(vectorstore, table_registry)

            self._q.put(("pdf_ok", {
                "qa_chain": qa_chain,
                "name":     Path(pdf_path).name,
                "tables":   len(table_registry),
                "chunks":   len(chunks),
            }))
        except Exception as exc:
            self._q.put(("error", str(exc)))

    def _answer_thread(self, question: str):
        try:
            answer = self.qa_chain.invoke(question)
            self._q.put(("answer", answer))
        except Exception as exc:
            self._q.put(("error", str(exc)))

    # ── 큐 폴링 (100 ms마다 메인 스레드에서 UI 업데이트) ─────────────────────────

    def _poll(self):
        try:
            while True:
                event, data = self._q.get_nowait()

                if event == "pdf_ok":
                    self.qa_chain = data["qa_chain"]
                    self.progress.stop()
                    self.upload_btn.config(state=tk.NORMAL)
                    self.entry.config(state=tk.NORMAL)
                    self.send_btn.config(state=tk.NORMAL)
                    self.status_var.set(
                        f"✓ {data['name']}  │  표 {data['tables']}개  │  청크 {data['chunks']}개"
                    )
                    self._system_line(f"{data['name']} 로드 완료 — 질문을 입력하세요")
                    self.entry.focus()

                elif event == "answer":
                    self.thinking_var.set("")
                    self._append_msg("bot", data)
                    self.entry.config(state=tk.NORMAL)
                    self.send_btn.config(state=tk.NORMAL)
                    self.entry.focus()

                elif event == "error":
                    self.progress.stop()
                    self.thinking_var.set("")
                    self.upload_btn.config(state=tk.NORMAL)
                    self.entry.config(state=tk.NORMAL)
                    self.send_btn.config(state=tk.NORMAL)
                    messagebox.showerror("오류", data, parent=self.root)

        except queue.Empty:
            pass

        self.root.after(100, self._poll)

    # ── 채팅창 헬퍼 ──────────────────────────────────────────────────────────────

    def _append_msg(self, role: str, text: str):
        self.chat.config(state=tk.NORMAL)
        if role == "user":
            self.chat.insert(tk.END, "\n나   ", "user_lbl")
            self.chat.insert(tk.END, text + "\n", "user_body")
        else:
            self.chat.insert(tk.END, "\n봇   ", "bot_lbl")
            self.chat.insert(tk.END, text + "\n", "bot_body")
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _system_line(self, text: str):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"\n── {text} ──\n", "system")
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)


# ── 진입점 ───────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    PDFChatbotApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
