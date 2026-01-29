"""
Document Upload Service for RAG
Uploads and processes documents for pgVector storage
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from database.operations import DatabaseOperations
from memory.embeddings import EmbeddingService
from rag import DocumentStore, SemanticChunker

logger = logging.getLogger(__name__)


class DocumentUploader:
    """
    Service for uploading documents to pgVector for RAG

    Supports:
    - PDF files
    - Plain text (TXT, MD, etc.)
    - Word documents (DOCX only - legacy .doc NOT supported)
    - HTML files
    - Automatic chunking and embedding
    - Batch processing
    - Progress tracking
    - Metadata extraction
    """

    def __init__(self, agent_id: str, document_store: Optional[DocumentStore] = None):
        self.agent_id = agent_id
        self.db_ops = DatabaseOperations()

        # Initialize document store if not provided
        if document_store:
            self.document_store = document_store
        else:
            from rag.data_wrangler import DataWrangler

            embedding_service = EmbeddingService()
            data_wrangler = DataWrangler(enable_dedup=True, min_quality_score=0.3)

            self.document_store = DocumentStore(
                database_ops=self.db_ops,
                embedding_service=embedding_service,
                chunking_strategy=SemanticChunker(
                    embedding_service=embedding_service,
                    similarity_threshold=config.SEMANTIC_SIMILARITY_THRESHOLD,
                ),
                data_wrangler=data_wrangler,
                kg_extractor=None,
            )

        logger.info(f"DocumentUploader initialized for agent: {agent_id}")

    def upload_file(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload a single file to the RAG system

        Args:
            file_path: Path to the file
            metadata: Optional metadata (author, tags, etc.)

        Returns:
            Upload result with statistics
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Uploading file: {file_path_obj.name}")

        # Read file content
        try:
            content = self._read_file(file_path_obj)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return {"success": False, "error": str(e), "file": str(file_path)}

        # Prepare metadata
        file_metadata = {
            "filename": file_path_obj.name,
            "file_type": file_path_obj.suffix.lower(),
            "file_size": file_path_obj.stat().st_size,
            "upload_date": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            **(metadata or {}),
        }

        # Store document (will automatically chunk and embed)
        try:
            result = self.document_store.upload_and_index(
                agent_id=self.agent_id,
                file_path=str(file_path_obj),
                file_content=content,
                metadata=file_metadata,
            )

            logger.info(
                f"File uploaded successfully: {file_path_obj.name} "
                f"({result.get('chunks_created', 0)} chunks)"
            )

            return {
                "success": True,
                "file": str(file_path),
                "chunks_created": result.get("chunks_created", 0),
                "document_id": result.get("document_id"),
                "metadata": file_metadata,
            }

        except Exception as e:
            logger.error(f"Failed to store document {file_path}: {e}")
            return {"success": False, "error": str(e), "file": str(file_path)}

    def upload_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Upload all files from a directory

        Args:
            directory_path: Path to directory
            recursive: Search subdirectories
            file_patterns: File patterns to include (e.g., ['*.pdf', '*.txt'])

        Returns:
            Batch upload results
        """
        directory = Path(directory_path)

        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")

        # Default patterns if none provided
        if file_patterns is None:
            file_patterns = [
                "*.pdf",
                "*.txt",
                "*.md",
                "*.docx",
            ]  # Note: .doc not supported

        # Collect files
        files = []
        for pattern in file_patterns:
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.glob(pattern))

        logger.info(f"Found {len(files)} files in {directory_path}")

        # Upload each file
        results = {
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "details": [],
        }

        for i, file_path in enumerate(files, 1):
            logger.info(f"Processing file {i}/{len(files)}: {file_path.name}")

            result = self.upload_file(file_path)
            results["details"].append(result)

            if result["success"]:
                results["successful"] += 1
                results["total_chunks"] += result.get("chunks_created", 0)
            else:
                results["failed"] += 1

        logger.info(
            f"Batch upload complete: {results['successful']}/{results['total_files']} successful, "
            f"{results['total_chunks']} chunks created"
        )

        return results

    def upload_text(
        self, text: str, title: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload raw text content

        Args:
            text: Text content
            title: Document title
            metadata: Optional metadata

        Returns:
            Upload result
        """
        logger.info(f"Uploading text: {title}")

        # Prepare metadata
        text_metadata = {
            "title": title,
            "source": "text_upload",
            "upload_date": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "content_length": len(text),
            **(metadata or {}),
        }

        try:
            # Use a dummy file path for text uploads
            result = self.document_store.upload_and_index(
                agent_id=self.agent_id,
                file_path=f"{title}.txt",
                file_content=text,
                metadata=text_metadata,
            )

            logger.info(
                f"Text uploaded successfully: {title} "
                f"({result.get('chunks_created', 0)} chunks)"
            )

            return {
                "success": True,
                "title": title,
                "chunks_created": result.get("chunks_created", 0),
                "document_id": result.get("document_id"),
                "metadata": text_metadata,
            }

        except Exception as e:
            logger.error(f"Failed to store text '{title}': {e}")
            return {"success": False, "error": str(e), "title": title}

    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document and its chunks from the system

        Args:
            document_id: Document ID to delete

        Returns:
            Success status
        """
        try:
            # Delete from document store (will cascade to chunks)
            self.document_store.delete_document(
                document_id=document_id, agent_id=self.agent_id
            )

            logger.info(f"Document deleted: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents for this agent

        Returns:
            List of document metadata
        """
        try:
            documents = self.document_store.list_documents(agent_id=self.agent_id)
            logger.info(f"Found {len(documents)} documents for agent {self.agent_id}")
            return documents
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    def _read_file(self, file_path: Path) -> str:
        """
        Read file content based on file type

        Args:
            file_path: Path to file

        Returns:
            File content as text
        """
        file_extension = file_path.suffix.lower()

        # Plain text files
        if file_extension in [
            ".txt",
            ".md",
            ".py",
            ".js",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
        ]:
            return self._read_text_file(file_path)

        # PDF files
        elif file_extension == ".pdf":
            return self._read_pdf(file_path)

        # Word documents (DOCX only)
        elif file_extension == ".docx":
            return self._read_docx(file_path)

        # Legacy .doc format (not supported)
        elif file_extension == ".doc":
            logger.error(
                f"Legacy .doc format not supported: {file_path.name}. "
                "Please convert to .docx using Microsoft Word or LibreOffice."
            )
            raise ValueError(
                f"Legacy .doc format not supported. Please convert '{file_path.name}' to .docx"
            )

        # HTML files
        elif file_extension in [".html", ".htm"]:
            return self._read_html(file_path)

        # Default: try as text
        else:
            logger.warning(
                f"Unknown file type {file_extension}, attempting to read as text"
            )
            return self._read_text_file(file_path)

    def _read_text_file(self, file_path: Path) -> str:
        """Read plain text file"""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _read_pdf(self, file_path: Path) -> str:
        """Read PDF file"""
        try:
            import PyPDF2

            logger.info(f"Reading PDF: {file_path.name}")
            text = []
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                logger.info(f"PDF has {num_pages} pages")

                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
                        logger.debug(f"Page {i + 1}: extracted {len(page_text)} chars")
                    else:
                        logger.warning(
                            f"Page {i + 1}: no text extracted (may be image-based PDF)"
                        )

            result = "\n\n".join(text)
            logger.info(
                f"PDF extraction complete: {len(result)} total characters from {len(text)} pages"
            )

            if len(result) == 0:
                logger.error(f"⚠️  PDF {file_path.name} resulted in ZERO text!")
                logger.error("   This PDF may be:")
                logger.error("   1. Image-based (scanned) - needs OCR")
                logger.error("   2. Encrypted/protected")
                logger.error("   3. Corrupted")

            return result

        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            raise ImportError("PyPDF2 required for PDF processing")

    def _read_docx(self, file_path: Path) -> str:
        """Read Word document"""
        try:
            import docx

            doc = docx.Document(str(file_path))
            text = []

            for paragraph in doc.paragraphs:
                text.append(paragraph.text)

            return "\n\n".join(text)

        except ImportError:
            logger.error(
                "python-docx not installed. Install with: pip install python-docx"
            )
            raise ImportError("python-docx required for DOCX processing")

    def _read_html(self, file_path: Path) -> str:
        """Read HTML file and extract text"""
        try:
            from bs4 import BeautifulSoup

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            return text

        except ImportError:
            logger.error(
                "beautifulsoup4 not installed. Install with: pip install beautifulsoup4"
            )
            raise ImportError("beautifulsoup4 required for HTML processing")


def upload_documents_cli():
    """
    Command-line interface for uploading documents
    """
    import argparse

    parser = argparse.ArgumentParser(description="Upload documents to RAG system")
    parser.add_argument("path", help="File or directory path")
    parser.add_argument("--agent-id", required=True, help="Agent ID")
    parser.add_argument(
        "--recursive", action="store_true", help="Process directories recursively"
    )
    parser.add_argument(
        "--patterns", nargs="+", help="File patterns (e.g., *.pdf *.txt)"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create uploader
    uploader = DocumentUploader(agent_id=args.agent_id)

    path = Path(args.path)

    if path.is_file():
        # Upload single file
        result = uploader.upload_file(str(path))
        if result["success"]:
            print(f"✓ Success: {result['chunks_created']} chunks created")
        else:
            print(f"✗ Failed: {result['error']}")

    elif path.is_dir():
        # Upload directory
        result = uploader.upload_directory(
            str(path), recursive=args.recursive, file_patterns=args.patterns
        )
        print(f"\nBatch Upload Results:")
        print(f"  Total files: {result['total_files']}")
        print(f"  Successful: {result['successful']}")
        print(f"  Failed: {result['failed']}")
        print(f"  Total chunks: {result['total_chunks']}")

    else:
        print(f"Error: Path not found: {path}")


if __name__ == "__main__":
    upload_documents_cli()
