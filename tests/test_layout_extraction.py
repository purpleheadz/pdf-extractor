import os
import sys
import unittest
import tempfile
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.extractor import extract_pdf_with_layout, process_layout_element

class TestLayoutExtraction(unittest.TestCase):
    """Tests specifically for the layout extraction functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test paths to sample PDFs."""
        # Create a test directory containing PDFs
        cls.test_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        cls.root_dir = cls.test_dir.parent
        cls.sample_pdfs_dir = cls.root_dir
        
        # Ensure we have test PDFs
        cls.test_pdfs = list(cls.sample_pdfs_dir.glob("*.pdf"))
        if not cls.test_pdfs:
            print("Warning: No test PDFs found. Tests will be skipped.")
        else:
            cls.test_pdf = str(cls.test_pdfs[0])  # Use the first PDF file for testing
    
    def test_extract_pdf_with_layout_headings(self):
        """Test that headings are properly detected in layout extraction."""
        if not self.test_pdfs:
            self.skipTest("No test PDFs available")
        
        # Extract with layout preservation
        markdown = extract_pdf_with_layout(self.test_pdf)
        
        # Check for heading markers
        self.assertIn("###", markdown)  # Should have at least some level-3 headings
        
        # Check for page structure
        self.assertIn("## Page", markdown)
        
        # Check for multiple paragraphs
        paragraphs = markdown.split("\n\n")
        self.assertTrue(len(paragraphs) > 5)  # Should have several paragraphs
    
    def test_extract_pdf_with_layout_tables(self):
        """Test that tables are properly extracted in layout mode."""
        if not self.test_pdfs:
            self.skipTest("No test PDFs available")
        
        # Extract with layout preservation
        markdown = extract_pdf_with_layout(self.test_pdf)
        
        # Look for table markers (not all PDFs may have tables)
        # We don't assert this is always true since not all test PDFs have tables
        if "| " in markdown and " |" in markdown:
            self.assertIn("#### Table", markdown)
    
    def test_extract_pdf_with_layout_page_specified(self):
        """Test extracting specific pages with layout preservation."""
        if not self.test_pdfs:
            self.skipTest("No test PDFs available")
            
        # Get page count
        with open(self.test_pdf, 'rb') as f:
            import PyPDF2
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
        
        if num_pages < 2:
            self.skipTest("PDF has too few pages for this test")
        
        # Extract just the first page
        markdown_page1 = extract_pdf_with_layout(self.test_pdf, pages="1")
        
        # Should have Page 1 but not Page 2
        self.assertIn("## Page 1", markdown_page1)
        self.assertNotIn("## Page 2", markdown_page1)
        
        # Extract two pages
        if num_pages >= 2:
            markdown_pages = extract_pdf_with_layout(self.test_pdf, pages="1-2")
            self.assertIn("## Page 1", markdown_pages)
            self.assertIn("## Page 2", markdown_pages)
    
    def test_layout_extraction_output_file(self):
        """Test that layout extraction correctly writes to an output file."""
        if not self.test_pdfs:
            self.skipTest("No test PDFs available")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp:
            temp_path = temp.name
        
        try:
            # Extract with layout preservation to the temp file
            extract_pdf_with_layout(self.test_pdf, temp_path)
            
            # Check the file was created and has content
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Verify content has expected format
            self.assertIn("# PDF Content", content)
            self.assertIn("## Page", content)
            self.assertTrue(len(content) > 100)  # Should have substantial content
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == '__main__':
    unittest.main()