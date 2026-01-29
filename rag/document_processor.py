"""
Document Processing
Handles PDF, TXT, MD, DOCX, HTML file ingestion with data wrangling
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .data_wrangler import DataWrangler

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process various document formats for RAG ingestion"""
    
    SUPPORTED_FORMATS = ['.pdf', '.txt', '.md', '.docx', '.html']
    
    def __init__(self, data_wrangler: Optional[DataWrangler] = None):
        """
        Initialize document processor
        
        Args:
            data_wrangler: DataWrangler instance for text cleaning
        """
        self.data_wrangler = data_wrangler or DataWrangler()
        self.pdf_available = self._check_pdf_support()
        self.docx_available = self._check_docx_support()
    
    def _check_pdf_support(self) -> bool:
        """Check if PDF processing is available"""
        try:
            import PyPDF2
            return True
        except ImportError:
            logger.warning("PyPDF2 not available. Install with: pip install PyPDF2")
            return False
    
    def _check_docx_support(self) -> bool:
        """Check if DOCX processing is available"""
        try:
            import docx
            return True
        except ImportError:
            logger.warning("python-docx not available. Install with: pip install python-docx")
            return False
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a file and extract text content with wrangling
        
        Args:
            file_path: Path to the file
        
        Returns:
            Dict with content, metadata, and wrangling results
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {extension}")
        
        # Extract text based on format
        if extension == '.pdf':
            raw_content = self._extract_pdf(path)
        elif extension == '.txt':
            raw_content = self._extract_txt(path)
        elif extension == '.md':
            raw_content = self._extract_txt(path)
        elif extension == '.docx':
            raw_content = self._extract_docx(path)
        elif extension == '.html':
            raw_content = self._extract_html(path)
        else:
            raw_content = ""
        
        # Apply data wrangling
        wrangled = self.data_wrangler.process(raw_content)
        
        metadata = {
            "filename": path.name,
            "extension": extension,
            "size_bytes": path.stat().st_size,
            "path": str(path.absolute()),
            "raw_length": len(raw_content),
            "cleaned_length": len(wrangled['cleaned_text']),
            "quality_score": wrangled['quality_score'],
            "num_tables": len(wrangled['tables']),
            "num_code_blocks": len(wrangled['code_blocks']),
            "num_lists": len(wrangled['lists'])
        }
        
        logger.info(
            f"Processed {path.name}: {len(raw_content)} → {len(wrangled['cleaned_text'])} chars, "
            f"quality={wrangled['quality_score']:.2f}"
        )
        
        return {
            "content": wrangled['cleaned_text'],
            "raw_content": raw_content,
            "wrangling_result": wrangled,
            "metadata": metadata,
            "success": True
        }
    
    def _extract_pdf(self, path: Path) -> str:
        """Extract text from PDF"""
        if not self.pdf_available:
            raise RuntimeError("PyPDF2 not installed")
        
        import PyPDF2
        
        text_parts = []
        
        try:
            with open(path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(f"[Page {page_num + 1}]\n{text}")
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise
        
        return "\n\n".join(text_parts)
    
    def _extract_txt(self, path: Path) -> str:
        """Extract text from TXT/MD files"""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1') as file:
                return file.read()
    
    def _extract_docx(self, path: Path) -> str:
        """Extract text from DOCX"""
        if not self.docx_available:
            raise RuntimeError("python-docx not installed")
        
        import docx
        
        doc = docx.Document(str(path))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        
        return "\n\n".join(paragraphs)
    
    def _extract_html(self, path: Path) -> str:
        """Extract text from HTML"""
        try:
            from bs4 import BeautifulSoup
            
            with open(path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                return text
        except ImportError:
            logger.warning("BeautifulSoup not available")
            return self._extract_txt(path)
    
