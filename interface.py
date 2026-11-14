import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
from rag_engine import extrair_texto_pdf, tratar_texto, criar_indice_invertido
from model_loader import carregar_modelo_llm, gerar_resposta

class RagPDFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG PDF Assistant")
        self.root.geometry("650x550")
        self.root.configure(bg="#f4f6f9")

        self.texto_completo = ""
        self.texto_tratado = ""
        self.modelo = None
        self.tokenizer = None
        self.index = None
        self.vetorizador = None

        self._build_ui()

    def _build_ui(self):
        # Header
        header = tk.Label(
            self.root, 
            text="📄 RAG PDF Assistant", 
            font=("Helvetica", 18, "bold"),
            bg="#f4f6f9",
            fg="#1f2937"
        )
        header.pack(pady=15)

        # Botão selecionar PDF
        self.btn_pdf = tk.Button(
            self.root,
            text="Selecionar Arquivo PDF",
            command=self.selecionar_pdf,
            font=("Helvetica", 11, "bold"),
            bg="#2563eb",
            fg="white",
            padx=15,
            pady=8,
            relief="flat",
            cursor="hand2"
        )
        self.btn_pdf.pack(pady=5)

        # Status do arquivo
        self.lbl_status = tk.Label(
            self.root,
            text="Nenhum arquivo carregado.",
            font=("Helvetica", 10),
            bg="#f4f6f9",
            fg="#6b7280"
        )
        self.lbl_status.pack(pady=5)

        # Campo de pergunta
        lbl_pergunta = tk.Label(
            self.root,
            text="Digite sua pergunta sobre o documento:",
            font=("Helvetica", 11),
            bg="#f4f6f9",
            fg="#374151"
        )
        lbl_pergunta.pack(pady=(15, 5))

        self.entry_pergunta = tk.Entry(self.root, width=60, font=("Helvetica", 11))
        self.entry_pergunta.pack(pady=5, ipady=4)
        self.entry_pergunta.bind("<Return>", lambda event: self.fazer_pergunta())

        # Botão Perguntar
        self.btn_perguntar = tk.Button(
            self.root,
            text="Perguntar ao Modelo",
            command=self.fazer_pergunta,
            font=("Helvetica", 11, "bold"),
            bg="#10b981",
            fg="white",
            padx=15,
            pady=6,
            relief="flat",
            cursor="hand2"
        )
        self.btn_perguntar.pack(pady=8)

        # Área de Resposta
        lbl_resposta = tk.Label(
            self.root,
            text="Resposta do Modelo:",
            font=("Helvetica", 11, "bold"),
            bg="#f4f6f9",
            fg="#374151"
        )
        lbl_resposta.pack(anchor="w", padx=40, pady=(10, 2))

        self.txt_resposta = scrolledtext.ScrolledText(
            self.root,
            width=65,
            height=10,
            font=("Helvetica", 10),
            wrap=tk.WORD,
            relief="solid",
            bd=1
        )
        self.txt_resposta.pack(pady=5)

    def selecionar_pdf(self):
        caminho = filedialog.askopenfilename(filetypes=[("Arquivos PDF", "*.pdf")])
        if not caminho:
            return

        self.lbl_status.config(text="Processando PDF e indexando texto...")
        self.root.update_idletasks()

        def processar():
            try:
                self.texto_completo = extrair_texto_pdf(caminho)
                self.texto_tratado = tratar_texto(self.texto_completo)
                self.index, self.vetorizador = criar_indice_invertido(self.texto_tratado)
                nome_arquivo = caminho.split("/")[-1].split("\\")[-1]
                self.lbl_status.config(text=f"PDF Carregado: {nome_arquivo} ({len(self.texto_completo)} caracteres)", fg="#059669")
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao ler o PDF: {e}")
                self.lbl_status.config(text="Erro ao carregar PDF.", fg="#dc2626")

        threading.Thread(target=processar, daemon=True).start()

    def fazer_pergunta(self):
        pergunta = self.entry_pergunta.get().strip()
        if not pergunta:
            messagebox.showwarning("Aviso", "Por favor, digite uma pergunta.")
            return

        if not self.texto_completo:
            messagebox.showwarning("Aviso", "Carregue um arquivo PDF primeiro.")
            return

        self.txt_resposta.delete("1.0", tk.END)
        self.txt_resposta.insert(tk.END, "Carregando modelo e gerando resposta... Aguarde.\n")
        self.root.update_idletasks()

        def gerar():
            try:
                if self.modelo is None:
                    self.modelo, self.tokenizer = carregar_modelo_llm()
                
                resposta = gerar_resposta(self.modelo, self.tokenizer, pergunta, self.texto_completo)
                self.txt_resposta.delete("1.0", tk.END)
                self.txt_resposta.insert(tk.END, resposta)
            except Exception as e:
                self.txt_resposta.delete("1.0", tk.END)
                self.txt_resposta.insert(tk.END, f"Erro ao gerar resposta: {e}\n\nVerifique se possui acesso ao modelo Hugging Face e dependências instaladas.")

        threading.Thread(target=gerar, daemon=True).start()

def main():
    root = tk.Tk()
    app = RagPDFApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
