import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    from gradio_client import Client, handle_file
except ImportError:
    print(" Erro: Biblioteca 'gradio_client' não encontrada.")
    print("   Por favor, instale usando: pip install -r requirements.txt")
    sys.exit(1)


@dataclass
class Config:
    PASTA_ENTRADA: Path = Path("imgs")
    PASTA_SAIDA: Path = Path("txts")
    
    EXTENSOES: set = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})
    
    MODELO: str = "SmilingWolf/wd-swinv2-tagger-v3"
    THRESHOLD_GERAL: float = 0.45
    THRESHOLD_PERSONAGEM: float = 0.85
    USAR_MCUT_GERAL: bool = False
    USAR_MCUT_PERSONAGEM: bool = False


class GeradorTags:

    def __init__(self):
        self.client = None
        self.stats = {"sucesso": 0, "falha": 0, "ignorado": 0}

    def iniciar(self):
        self._exibir_boas_vindas()
        
        if not self._verificar_pastas():
            return

        imagens = self._listar_imagens()
        if not imagens:
            print(f" Nenhuma imagem encontrada na pasta '{Config.PASTA_ENTRADA}'.")
            return

        self._conectar_api()
        if not self.client:
            return

        self._processar_lote(imagens)
        self._exibir_relatorio_final(len(imagens))

    def _exibir_boas_vindas(self):
        print("\n" + "="*50)
        print("  GERADOR DE TAGS AUTOMÁTICO v1.0")
        print("="*50 + "\n")

    def _verificar_pastas(self) -> bool:
        if not Config.PASTA_ENTRADA.exists():
            print(f" Pasta de entrada não encontrada: {Config.PASTA_ENTRADA}")
            print("   -> Crie a pasta e coloque suas imagens nela.")
            return False

        if not Config.PASTA_SAIDA.exists():
            Config.PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
            print(f" Pasta de saída criada: {Config.PASTA_SAIDA}")
        
        return True

    def _listar_imagens(self) -> List[Path]:
        return sorted([
            arq for arq in Config.PASTA_ENTRADA.iterdir() 
            if arq.is_file() and arq.suffix.lower() in Config.EXTENSOES
        ])

    def _conectar_api(self):
        print(" Conectando à IA (pode levar alguns segundos)...", end=" ", flush=True)
        try:
            self.client = Client("SmilingWolf/wd-tagger")
            print(" Conectado!")
        except Exception as e:
            print(f"\n Falha na conexão: {e}")
            print("   Verifique sua internet ou tente novamente mais tarde.")

    def _processar_lote(self, imagens: List[Path]):
        total = len(imagens)
        inicio = time.time()
        
        print(f"\n Iniciando processamento de {total} imagens...\n")

        for disponivel, imagem in enumerate(imagens, 1):
            caminho_txt = Config.PASTA_SAIDA / f"{imagem.stem}.txt"
            
            if caminho_txt.exists():
                print(f"  [{disponivel}/{total}] Pulando {imagem.name} (já existe)")
                self.stats["ignorado"] += 1
                continue

            sucesso = self._gerar_e_salvar(imagem, caminho_txt, disponivel, total)
            if sucesso:
                self.stats["sucesso"] += 1
            else:
                self.stats["falha"] += 1

        self.stats["tempo_total"] = time.time() - inicio

    def _gerar_e_salvar(self, imagem: Path, destino: Path, atual: int, total: int) -> bool:
        print(f" [{atual}/{total}] Analisando: {imagem.name}...", end=" ", flush=True)
        
        try:
            resultado = self.client.predict(
                image=handle_file(str(imagem.absolute())),
                model_repo=Config.MODELO,
                general_thresh=Config.THRESHOLD_GERAL,
                general_mcut_enabled=Config.USAR_MCUT_GERAL,
                character_thresh=Config.THRESHOLD_PERSONAGEM,
                character_mcut_enabled=Config.USAR_MCUT_PERSONAGEM,
                api_name="/predict"
            )
            
            tags = resultado[0]
            
            with open(destino, "w", encoding="utf-8") as f:
                f.write(tags)
            
            print(" Feito!")
            return True

        except Exception as e:
            print(f"\n Erro ao processar {imagem.name}: {e}")
            self._log_erro(imagem.name, str(e))
            return False

    def _log_erro(self, arquivo: str, erro: str):
        log_path = Config.PASTA_SAIDA / "_erros.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {arquivo}: {erro}\n")

    def _exibir_relatorio_final(self, total_arquivos: int):
        tempo = self.stats.get("tempo_total", 0)
        media = tempo / max(total_arquivos - self.stats["ignorado"], 1)

        print("\n" + "-"*50)
        print(" RELATORIO FINAL")
        print("-"*50)
        print(f" Processados com sucesso: {self.stats['sucesso']}")
        print(f"  Já existiam (ignorados): {self.stats['ignorado']}")
        print(f" Falhas: {self.stats['falha']}")
        print(f" Tempo total: {tempo:.2f}s")
        if self.stats["sucesso"] > 0:
            print(f" Média por imagem: {media:.2f}s")
        print("-"*50 + "\n")


if __name__ == "__main__":
    app = GeradorTags()
    app.iniciar()
