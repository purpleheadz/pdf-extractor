"""Tests for the PDF extractor module."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.extractor import (
    extract_images_from_pdf,
    extract_pdf_as_markdown,
    extract_pdf_with_layout,
    extract_tables_as_markdown,
    extract_text_from_pdf,
)


class TestExtractor(unittest.TestCase):
    """
    Tests for the PDF extractor functions.

    Note: These tests require sample PDF files in the test_pdfs directory.
    """

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

    def test_extract_text_from_pdf(self):
        """Test extracting text from PDF."""
        if not self.test_pdfs:
            self.skipTest("No test PDFs available")

        # Test with direct return
        text = extract_text_from_pdf(self.test_pdf)
        self.assertIsNotNone(text)
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

        # Test with file output
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
            temp_path = temp.name

        try:
            extract_text_from_pdf(self.test_pdf, temp_path)
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertTrue(len(content) > 0)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_extract_tables_as_markdown(self):
        """Test extracting tables as markdown."""
        if not self.test_pdfs:
            self.skipTest("No test PDFs available")

        # Test with direct return
        markdown = extract_tables_as_markdown(self.test_pdf)
        # Tables might not be in all PDFs, so we don't assert length > 0
        self.assertIsInstance(markdown, str)

        # Test with file output
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as temp:
            temp_path = temp.name

        try:
            extract_tables_as_markdown(self.test_pdf, temp_path)
            self.assertTrue(os.path.exists(temp_path))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_extract_pdf_as_markdown(self):
        """Test extracting PDF as markdown with integrated tables."""
        if not self.test_pdfs:
            self.skipTest("No test PDFs available")

        # Test with direct return
        markdown = extract_pdf_as_markdown(self.test_pdf)
        self.assertIsNotNone(markdown)
        self.assertIsInstance(markdown, str)
        self.assertTrue(len(markdown) > 0)

        # Test with file output
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as temp:
            temp_path = temp.name

        try:
            extract_pdf_as_markdown(self.test_pdf, temp_path)
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertTrue(len(content) > 0)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_extract_pdf_with_layout(self):
        """Test extracting PDF with layout preservation."""
        if not self.test_pdfs:
            self.skipTest("No test PDFs available")

        # Test with direct return
        markdown = extract_pdf_with_layout(self.test_pdf)
        self.assertIsNotNone(markdown)
        self.assertIsInstance(markdown, str)
        self.assertTrue(len(markdown) > 0)

        # Test with file output
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as temp:
            temp_path = temp.name

        try:
            extract_pdf_with_layout(self.test_pdf, temp_path)
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertTrue(len(content) > 0)

            # Check for layout-specific markers
            self.assertIn("# PDF Content", content)
            self.assertIn("## Page", content)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_page_specific_extraction(self):
        """Test extracting specific pages from PDF."""
        if not self.test_pdfs:
            self.skipTest("No test PDFs available")

        # Test with specific page
        markdown = extract_pdf_as_markdown(self.test_pdf, pages="1")
        self.assertIsNotNone(markdown)
        self.assertIsInstance(markdown, str)

        # Test with page range
        if len(self.test_pdfs) > 0:
            # Get page count from the PDF
            with open(self.test_pdf, "rb") as f:
                import PyPDF2

                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)

            if num_pages > 1:
                markdown = extract_pdf_as_markdown(self.test_pdf, pages="1-2")
                self.assertIsNotNone(markdown)
                self.assertIsInstance(markdown, str)

    def test_extract_images_from_pdf(self):
        """Test extracting images from PDF."""
        if not self.test_pdfs:
            self.skipTest("No test PDFs available")

        # Create a temp directory for extracted images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test image extraction
            image_paths = extract_images_from_pdf(self.test_pdf, output_dir=temp_dir)

            # Images might not exist in all PDFs, so we cannot assert the length > 0
            self.assertIsInstance(image_paths, list)

            # Check if images were saved to the directory if any were found
            if image_paths:
                self.assertTrue(os.path.exists(temp_dir))
                self.assertTrue(os.path.exists(image_paths[0]))

                # Check if image files are not empty
                for image_path in image_paths:
                    self.assertTrue(os.path.getsize(image_path) > 0)

    def test_pdf_to_markdown_with_images(self):
        """Test extracting PDF as markdown with images."""
        if not self.test_pdfs:
            self.skipTest("No test PDFs available")

        # Create a temp directory for extracted images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with image extraction enabled
            markdown = extract_pdf_as_markdown(
                self.test_pdf, extract_images=True, image_dir=temp_dir
            )
            self.assertIsNotNone(markdown)
            self.assertIsInstance(markdown, str)
            self.assertTrue(len(markdown) > 0)

            # Test with image extraction disabled
            markdown_no_images = extract_pdf_as_markdown(
                self.test_pdf, extract_images=False
            )
            self.assertIsNotNone(markdown_no_images)
            self.assertIsInstance(markdown_no_images, str)
            self.assertTrue(len(markdown_no_images) > 0)

            # Test layout extraction with images
            markdown_layout = extract_pdf_with_layout(
                self.test_pdf, extract_images=True
            )
            self.assertIsNotNone(markdown_layout)
            self.assertIsInstance(markdown_layout, str)
            self.assertTrue(len(markdown_layout) > 0)

            # Test layout extraction without images
            markdown_layout_no_images = extract_pdf_with_layout(
                self.test_pdf, extract_images=False
            )
            self.assertIsNotNone(markdown_layout_no_images)
            self.assertIsInstance(markdown_layout_no_images, str)
            self.assertTrue(len(markdown_layout_no_images) > 0)


if __name__ == "__main__":
    unittest.main()
