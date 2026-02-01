"""
Script para fazer upload dos documentos RAG
Processa dw-grpo.pdf e outros PDFs na pasta sample/docs/rag
"""

import logging
from utils.logging_config import setup_logging
from services.document_uploader import DocumentUploader

setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


def main():
    """Upload documentos RAG para o agente"""
    
    print("=" * 60)
    print("Upload de Documentos RAG")
    print("=" * 60)
    print()
    
    # IMPORTANTE: Use o mesmo agent_id que será usado no main.py
    agent_id = "user_demo_agent"
    
    print(f"Agent ID: {agent_id}")
    print(f"Diretório: sample/docs/rag")
    print()
    
    uploader = DocumentUploader(agent_id=agent_id)
    
    # Upload da pasta RAG (inclui dw-grpo.pdf, RAG-PAPER.pdf, system1-system2.pdf)
    print("Processando documentos...")
    result = uploader.upload_directory(
        directory_path="C:\\Users\\gabri\\Desktop\\memGPT\\sample\\docs\\rag",
        recursive=True,
        file_patterns=['*.pdf', '*.txt', '*.md']
    )
    
    # Exibir resultados
    print("\n" + "=" * 60)
    print("RESULTADO DO UPLOAD")
    print("=" * 60)
    print(f"✓ Arquivos processados: {result['successful']}/{result['total_files']}")
    print(f"✓ Total de chunks: {result['total_chunks']}")
    print()
    
    if result['successful'] > 0:
        print("Documentos indexados com sucesso!")
        print("\nDetalhes por arquivo:")
        for detail in result['details']:
            if detail['success']:
                from pathlib import Path
                print(f"  ✓ {Path(detail['file']).name}: {detail.get('chunks_created', 0)} chunks")
            else:
                from pathlib import Path
                print(f"  ✗ {Path(detail['file']).name}: {detail.get('error', 'Unknown error')}")
    
    if result.get('failed'):
        print(f"\n✗ Falhas: {result['failed']}")

    print("\n✓ Upload concluído!")
    print(f"✓ Os documentos foram indexados para o agente: {agent_id}")
    
    # VERIFICAÇÃO CRÍTICA: Contar chunks na tabela
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO DA TABELA document_chunks")
    print("=" * 60)
    try:
        from database.connection import db
        with db.get_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks WHERE agent_id = %s", (agent_id,))
            chunk_count = cur.fetchone()[0]
            print(f"Chunks encontrados na tabela: {chunk_count}")
            
            if chunk_count == 0:
                print("\n⚠️  PROBLEMA DETECTADO: Tabela está vazia após upload!")
                print("Possíveis causas:")
                print("  1. Commits não estão sendo executados")
                print("  2. Tabela foi recriada após upload (problema de dimensão)")
                print("  3. Todos os embeddings foram rejeitados (NaN/Inf)")
                print("\nVerifique os logs acima para mais detalhes.")
            elif chunk_count != result['total_chunks']:
                print(f"\n⚠️  AVISO: Esperado {result['total_chunks']} chunks, encontrado {chunk_count}")
                print(f"Diferença: {result['total_chunks'] - chunk_count} chunks faltando")
            else:
                print(f"\n✓ SUCESSO! Todos os {chunk_count} chunks foram armazenados corretamente!")
    except Exception as e:
        print(f"✗ Erro ao verificar tabela: {e}")
    
    print("\nAgora você pode executar queries no main.py")


if __name__ == "__main__":
    main()