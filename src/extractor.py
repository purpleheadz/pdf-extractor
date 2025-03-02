"""PDF content extractor for converting PDFs to text, markdown, and extracting images."""

import argparse
import os
import re
import traceback
from io import StringIO

import PyPDF2
import pandas as pd
import tabula
import tabulate
from PIL import Image
from pdfminer.high_level import extract_pages, extract_text as pdfminer_extract_text
from pdfminer.layout import (
    LAParams,
    LTChar,
    LTFigure,
    LTImage,
    LTTextBox,
    LTTextContainer,
    LTTextLine,
)


def extract_text_from_pdf(pdf_path, output_path=None):
    """
    Extract text from a PDF file and optionally save to a text file.

    Args:
        pdf_path (str): Path to the PDF file
        output_path (str, optional): Path to save the extracted text

    Returns:
        str: Extracted text from PDF
    """
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""

            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
                text += "\n\n--- Page {} End ---\n\n".format(page_num + 1)

            if output_path:
                with open(output_path, "w", encoding="utf-8") as output_file:
                    output_file.write(text)
                print(f"Extracted text saved to {output_path}")

            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None


def extract_tables_as_markdown(pdf_path, output_path=None, pages="all"):
    """
    Extract tables from a PDF file and convert them to markdown format.

    Args:
        pdf_path (str): Path to the PDF file
        output_path (str, optional): Path to save the extracted tables as markdown
        pages (str or list): Page numbers to extract tables from (default: 'all')

    Returns:
        str: Markdown formatted tables
    """
    try:
        # Parse pages parameter if it's a string
        if isinstance(pages, str) and pages != "all":
            pages_list = []
            for part in pages.split(","):
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    pages_list.extend([str(p) for p in range(start, end + 1)])
                else:
                    pages_list.append(part)
            pages = pages_list

        # Try multiple methods to extract tables
        table_extraction_methods = [
            {"lattice": True, "stream": False},
            {"lattice": False, "stream": True},
            {"lattice": True, "stream": True},
        ]

        tables = []

        for method in table_extraction_methods:
            try:
                # Read tables from PDF
                extracted_tables = tabula.read_pdf(
                    pdf_path,
                    pages=pages,
                    multiple_tables=True,
                    lattice=method["lattice"],
                    stream=method["stream"],
                    guess=True,
                    pandas_options={"header": None},
                )

                if extracted_tables and len(extracted_tables) > 0:
                    print(
                        f"Found {len(extracted_tables)} tables using method: lattice={method['lattice']}, stream={method['stream']}"
                    )

                    # Process each table
                    for i, df in enumerate(extracted_tables):
                        if not df.empty and df.shape[0] > 0 and df.shape[1] > 0:
                            cleaned_df = clean_dataframe(df)
                            if not cleaned_df.empty:
                                tables.append(cleaned_df)

                    # If we found tables, break out of the loop
                    if tables:
                        break
            except Exception as e:
                print(f"Table extraction method failed: {e}")

        if not tables:
            print("No tables found in the PDF.")
            return ""

        # Convert tables to markdown
        markdown_output = ""
        for i, df in enumerate(tables):
            markdown_output += f"## Table {i+1}\n\n"

            try:
                # Convert dataframe to markdown
                table_md = df.to_markdown(index=False)
                markdown_output += table_md + "\n\n"
            except Exception as e:
                # Fallback if to_markdown fails
                print(f"Error converting table to markdown: {e}")
                try:
                    # Try alternative approach with tabulate directly
                    table_md = tabulate.tabulate(
                        df.values, headers=df.columns, tablefmt="pipe"
                    )
                    markdown_output += table_md + "\n\n"
                except Exception as e2:
                    print(f"Second attempt at table conversion failed: {e2}")
                    # Last resort fallback
                    markdown_output += "```\n" + df.to_string(index=False) + "\n```\n\n"

        if output_path:
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(markdown_output)
            print(f"Extracted tables saved to {output_path}")

        return markdown_output
    except Exception as e:
        print(f"Error extracting tables from PDF: {e}")
        import traceback

        traceback.print_exc()
        return None


def extract_pdf_as_markdown(
    pdf_path,
    output_path=None,
    pages="all",
    extract_images=True,
    image_dir="extracted_images",
):
    """
    Extract entire PDF content as markdown with tables integrated into the content flow.

    Args:
        pdf_path (str): Path to the PDF file
        output_path (str, optional): Path to save the extracted content as markdown
        pages (str or list): Page numbers to extract from (default: 'all')
        extract_images (bool): Whether to extract images from PDF (default: True)
        image_dir (str): Directory to save extracted images

    Returns:
        str: Markdown formatted content with tables properly integrated
    """
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        num_pages = len(reader.pages)

        # Extract images if requested
        image_paths = []
        if extract_images:
            print("Extracting images from PDF...")
            image_paths = extract_images_from_pdf(pdf_path, output_dir=image_dir)
            print(f"Extracted {len(image_paths)} images to {image_dir}/")

        # Parse pages parameter
        if pages == "all":
            pages_to_process = range(1, num_pages + 1)
        else:
            # Parse pages parameter
            if isinstance(pages, str):
                pages_list = []
                for part in pages.split(","):
                    if "-" in part:
                        start, end = map(int, part.split("-"))
                        pages_list.extend(range(start, end + 1))
                    else:
                        pages_list.append(int(part))
                pages_to_process = pages_list
            else:
                pages_to_process = pages

        # Format pages for tabula (1-indexed)
        if pages == "all":
            tabula_pages = "all"
        else:
            tabula_pages = [str(p) for p in pages_to_process]

        print(f"Extracting tables from pages: {tabula_pages}")

        # Extract tables from PDF with page information
        tables_with_page_info = []

        # Try multiple methods to extract tables
        table_extraction_methods = [
            {"lattice": True, "stream": False},
            {"lattice": False, "stream": True},
            {"lattice": True, "stream": True},
        ]

        for method in table_extraction_methods:
            try:
                dfs = tabula.read_pdf(
                    pdf_path,
                    pages=tabula_pages,
                    multiple_tables=True,
                    lattice=method["lattice"],
                    stream=method["stream"],
                    guess=True,
                    pandas_options={"header": None},
                )

                if dfs and len(dfs) > 0:
                    print(
                        f"Found {len(dfs)} tables using method: lattice={method['lattice']}, stream={method['stream']}"
                    )

                    # Get page information for each table
                    page_data = tabula.read_pdf(
                        pdf_path,
                        pages=tabula_pages,
                        multiple_tables=True,
                        lattice=method["lattice"],
                        stream=method["stream"],
                        output_format="json",
                    )

                    for i, df in enumerate(dfs):
                        try:
                            # Try to get page number from JSON output
                            if i < len(page_data) and "page" in page_data[i]:
                                page_num = page_data[i]["page"]
                            elif i < len(page_data) and "page_number" in page_data[i]:
                                page_num = page_data[i]["page_number"]
                            else:
                                # If unable to determine page number, assign to first page
                                # This is a fallback and not ideal
                                page_num = (
                                    list(pages_to_process)[0] if pages_to_process else 1
                                )

                            # Only add non-empty tables
                            if not df.empty and df.shape[0] > 0 and df.shape[1] > 0:
                                cleaned_df = clean_dataframe(df)
                                if not cleaned_df.empty:
                                    tables_with_page_info.append((page_num, cleaned_df))
                        except Exception as e:
                            print(f"Error processing table {i}: {e}")

                    # If we found tables, break out of the loop
                    if tables_with_page_info:
                        break
            except Exception as e:
                print(f"Table extraction method failed: {e}")

        if not tables_with_page_info:
            print("No tables found in the PDF.")
        else:
            print(f"Successfully extracted {len(tables_with_page_info)} tables")

        # Extract text with pdfminer for better structure preservation
        text = pdfminer_extract_text(pdf_path, laparams=LAParams())

        # Convert text to markdown
        markdown_text = convert_text_to_markdown(text)

        # Initialize markdown output
        markdown_output = "# PDF Content\n\n"

        # Add image references to markdown if images were extracted
        if image_paths:
            markdown_output += "## Extracted Images\n\n"
            for i, image_path in enumerate(image_paths):
                markdown_output += f"![Image {i+1}]({image_path})\n\n"

        # Split text by page markers
        sections = markdown_text.split("--- Page")
        markdown_output += sections[0].strip()  # Add content before first page marker

        # Group tables by page
        tables_by_page = {}
        for page_num, df in tables_with_page_info:
            if page_num not in tables_by_page:
                tables_by_page[page_num] = []
            tables_by_page[page_num].append(df)

        # Process each page and integrate tables
        for i, page_num in enumerate(pages_to_process):
            if i < len(sections) - 1:
                page_content = sections[i + 1]

                # Extract page header and content
                if "---" in page_content:
                    header, content = page_content.split("---", 1)
                    header = header.strip()
                else:
                    header = str(page_num)
                    content = page_content

                # Add page header
                markdown_output += f"\n\n## Page {header}\n\n"

                # Add tables for this page integrated with content
                if page_num in tables_by_page and tables_by_page[page_num]:
                    # Add a paragraph about tables being found
                    content_parts = content.split("\n\n")

                    # Add the first few paragraphs
                    for j, part in enumerate(content_parts[:2]):
                        if part.strip():
                            markdown_output += part.strip() + "\n\n"

                    # Add tables in the middle
                    for k, df in enumerate(tables_by_page[page_num]):
                        table_title = f"Table {k+1}"

                        # Add a small header for the table
                        markdown_output += f"### {table_title}\n\n"

                        try:
                            # Convert dataframe to markdown using tabulate
                            table_md = df.to_markdown(index=False)
                            markdown_output += table_md + "\n\n"
                        except Exception as e:
                            # Fallback if to_markdown fails
                            print(f"Error converting table to markdown: {e}")
                            try:
                                # Try alternative approach with tabulate directly
                                table_md = tabulate.tabulate(
                                    df.values, headers=df.columns, tablefmt="pipe"
                                )
                                markdown_output += table_md + "\n\n"
                            except Exception as e2:
                                print(
                                    f"Second attempt at table conversion failed: {e2}"
                                )
                                # Last resort fallback
                                markdown_output += (
                                    "```\n" + df.to_string(index=False) + "\n```\n\n"
                                )

                    # Add remaining content
                    for part in content_parts[2:]:
                        if part.strip():
                            markdown_output += part.strip() + "\n\n"
                else:
                    # Just add the content if no tables
                    markdown_output += content.strip() + "\n\n"

        if output_path:
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(markdown_output)
            print(f"Extracted markdown saved to {output_path}")

        return markdown_output
    except Exception as e:
        print(f"Error extracting PDF as markdown: {e}")
        import traceback

        traceback.print_exc()
        return None


def clean_dataframe(df):
    """
    Clean the dataframe by removing empty rows and columns,
    handling NaN values, and improving table structure.
    Propagates merged cell values to all relevant rows and columns.
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()

    # Fill NaN values with empty string
    df = df.fillna("")

    # Clean column names if they are all numbers or NaN
    if all(isinstance(col, (int, float)) or pd.isna(col) for col in df.columns):
        df.columns = [f"Column {i+1}" for i in range(len(df.columns))]

    # Remove empty rows where all values are empty strings or NaN
    df = df.loc[~(df == "").all(axis=1)]

    # Remove empty columns
    df = df.loc[:, ~(df == "").all(axis=0)]

    # Create a copy of the dataframe to store the processed values
    processed_df = df.copy()

    # Store original values before any processing for reference
    original_values = df.copy()

    # Step 1: Process vertical merges (cells merged across multiple rows)
    for col in processed_df.columns:
        prev_value = None
        for i in range(len(processed_df)):
            current_value = processed_df.at[i, col]
            # If cell is empty and we have a previous value, it's likely a merged cell
            if current_value == "" and prev_value is not None:
                processed_df.at[i, col] = prev_value
            elif current_value != "":
                prev_value = current_value

    # Step 2: Process horizontal merges (cells merged across multiple columns)
    for i in range(len(processed_df)):
        # For each row, check for empty cells that might be part of horizontal merges
        for j, col in enumerate(processed_df.columns):
            # Skip if the cell wasn't originally empty or if it's the first column
            if original_values.at[i, col] != "" or j == 0:
                continue

            # Look for the nearest non-empty cell to the left in the original data
            left_value = None
            for k in range(j - 1, -1, -1):
                left_col = processed_df.columns[k]
                # Use the value from the original dataframe to check if it was empty originally
                if original_values.at[i, left_col] != "":
                    left_value = processed_df.at[
                        i, left_col
                    ]  # Get the potentially updated value
                    break

            # If we found a non-empty cell to the left, and current cell is still empty
            # (meaning it wasn't filled by vertical merge processing), fill it
            if left_value is not None and processed_df.at[i, col] == "":
                processed_df.at[i, col] = left_value

    # Convert all values to strings
    for col in processed_df.columns:
        processed_df[col] = processed_df[col].astype(str)

    return processed_df


def convert_text_to_markdown(text):
    """
    Convert extracted text to markdown format.
    """
    # Replace multiple newlines with double newlines for markdown paragraphs
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Try to identify headers by looking for short lines followed by empty lines
    lines = text.split("\n")
    result = []

    for i, line in enumerate(lines):
        line = line.strip()
        if line:
            if len(line) < 100 and i < len(lines) - 1 and not lines[i + 1].strip():
                # This looks like it might be a header
                if (
                    i > 0 and not lines[i - 1].strip()
                ):  # Check if there's an empty line before
                    if line.isupper():  # All caps might be a main header
                        result.append(f"## {line}")
                    else:
                        result.append(f"### {line}")
                else:
                    result.append(line)
            else:
                result.append(line)
        else:
            result.append("")

    return "\n".join(result)


def extract_images_from_pdf(pdf_path, output_dir="extracted_images"):
    """
    Extract all images from PDF file and save them to a directory.

    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save extracted images

    Returns:
        list: List of saved image paths
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Open the PDF file
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            image_paths = []

            # Extract images from each page
            for page_num, page in enumerate(reader.pages):

                # Get XObject resources (contain images)
                if "/Resources" in page and "/XObject" in page["/Resources"]:
                    x_objects = page["/Resources"]["/XObject"]

                    # Extract each image from resources
                    for obj_name, obj in x_objects.items():
                        # Check if this is an image object by getting the object
                        # Sometimes PyPDF2 returns IndirectObject which needs to be resolved
                        try:
                            if isinstance(obj, PyPDF2.generic.IndirectObject):
                                obj = obj.get_object()

                            if (
                                obj.get("/Type") == "/XObject"
                                and obj.get("/Subtype") == "/Image"
                            ):
                                try:
                                    # Generate unique filename
                                    image_hash = hash(obj_name) & 0xFFFFFFFF
                                    image_filename = f"{output_dir}/page{page_num+1}_image_{image_hash}.png"

                                    # Get image data
                                    if "/Filter" in obj:
                                        filter_type = obj["/Filter"]

                                    # Handle different filter types
                                    if (
                                        isinstance(filter_type, list)
                                        and "/DCTDecode" in filter_type
                                        or filter_type == "/DCTDecode"
                                    ):
                                        # JPEG image
                                        try:
                                            image_data = obj.get_data()
                                            with open(
                                                image_filename.replace(".png", ".jpg"),
                                                "wb",
                                            ) as img_file:
                                                img_file.write(image_data)
                                            image_paths.append(
                                                image_filename.replace(".png", ".jpg")
                                            )
                                        except Exception as e:
                                            print(f"Error saving JPEG image: {e}")

                                    elif (
                                        isinstance(filter_type, list)
                                        and "/FlateDecode" in filter_type
                                        or filter_type == "/FlateDecode"
                                    ):
                                        # PNG or other formats that need conversion
                                        try:
                                            width = obj.get("/Width")
                                            height = obj.get("/Height")
                                            color_space = obj.get(
                                                "/ColorSpace", "/DeviceRGB"
                                            )
                                            bits_per_component = obj.get(
                                                "/BitsPerComponent", 8
                                            )

                                            # Get image data and convert
                                            try:
                                                image_data = obj.get_data()

                                                # Create PIL Image from data
                                                if color_space == "/DeviceRGB":
                                                    mode = "RGB"
                                                elif color_space == "/DeviceGray":
                                                    mode = "L"
                                                else:
                                                    mode = "RGB"  # Fallback

                                                # Convert the raw image data
                                                # For some PDFs, this might need specific handling
                                                try:
                                                    image = Image.frombytes(
                                                        mode,
                                                        (width, height),
                                                        image_data,
                                                    )
                                                    image.save(image_filename)
                                                    image_paths.append(image_filename)
                                                except Exception as e:
                                                    print(
                                                        f"Image conversion error: {e}"
                                                    )

                                            except Exception as e:
                                                print(
                                                    f"Error processing FlateDecode image: {e}"
                                                )

                                        except Exception as e:
                                            print(
                                                f"Error getting image attributes: {e}"
                                            )

                                    elif obj.get("/Filter") is not None:
                                        # Other filter types
                                        try:
                                            image_data = obj.get_data()
                                            with open(image_filename, "wb") as img_file:
                                                img_file.write(image_data)
                                            image_paths.append(image_filename)
                                        except Exception as e:
                                            print(
                                                f"Error saving image with filter {filter_type}: {e}"
                                            )

                                    else:
                                        # Raw image data
                                        try:
                                            image_data = obj.get_data()
                                            with open(image_filename, "wb") as img_file:
                                                img_file.write(image_data)
                                            image_paths.append(image_filename)
                                        except Exception as e:
                                            print(f"Error saving raw image data: {e}")
                                except Exception as e:
                                    print(f"Error extracting image from PDF: {e}")
                        except Exception as e:
                            # Skip objects that can't be processed
                            pass

            return image_paths

    except Exception as e:
        print(f"Error extracting images from PDF: {e}")
        import traceback

        traceback.print_exc()
        return []


def extract_pdf_with_layout(
    pdf_path, output_path=None, pages="all", extract_images=True
):
    """
    Extract PDF content preserving layout information by directly processing
    PDF layout elements and converting them to appropriate markdown.

    Args:
        pdf_path (str): Path to the PDF file
        output_path (str, optional): Path to save the extracted content as markdown
        pages (str or list): Page numbers to extract from (default: 'all')
        extract_images (bool): Whether to extract and save images (default: True)

    Returns:
        str: Markdown formatted content with layout preserved
    """
    try:
        # Parse pages parameter
        if pages == "all":
            page_numbers = None  # Process all pages
        else:
            # Convert page specification to a list of page numbers
            if isinstance(pages, str):
                page_list = []
                for part in pages.split(","):
                    if "-" in part:
                        start, end = map(int, part.split("-"))
                        page_list.extend(range(start, end + 1))
                    else:
                        page_list.append(int(part))
                page_numbers = page_list
            else:
                page_numbers = pages

        # Setup layout parameters
        la_params = LAParams(
            line_margin=0.5,
            char_margin=2.0,
            word_margin=0.1,
            boxes_flow=0.5,
            detect_vertical=True,
        )

        # Extract images if requested
        image_paths = []
        image_dir = "extracted_images"
        if extract_images:
            print("Extracting images from PDF...")
            image_paths = extract_images_from_pdf(pdf_path, output_dir=image_dir)
            print(f"Extracted {len(image_paths)} images to {image_dir}/")

        # Initialize markdown output
        markdown_content = "# PDF Content\n\n"

        # Add image references to markdown if images were extracted
        if image_paths:
            markdown_content += "## Extracted Images\n\n"
            for i, image_path in enumerate(image_paths):
                markdown_content += f"![Image {i+1}]({image_path})\n\n"

        # Process pages with extract_pages
        page_layouts = list(extract_pages(pdf_path, laparams=la_params))

        # Filter pages if needed
        if page_numbers is not None:
            # Convert to 0-based indexing for extract_pages
            page_indices = [n - 1 for n in page_numbers]
            page_layouts = [
                page_layouts[i] for i in page_indices if i < len(page_layouts)
            ]

        # Process each page
        for page_num, layout in enumerate(page_layouts):
            # Add page header
            markdown_content += f"## Page {page_num + 1}\n\n"

            # Dictionary to track text positions
            text_by_y = {}

            # First pass: collect all text elements by y-position
            # This helps maintain the reading order based on vertical position
            for element in layout:
                process_layout_element(element, text_by_y)

            # Sort text elements by y-position (top to bottom)
            sorted_ys = sorted(
                text_by_y.keys(), reverse=True
            )  # Reverse for top-to-bottom

            # Second pass: Generate markdown from collected text elements
            for y_pos in sorted_ys:
                line_elements = text_by_y[y_pos]

                # Sort elements from left to right
                line_elements.sort(key=lambda x: x[0])

                # Check if this might be a heading (determined by size, boldness, etc.)
                is_heading = False
                text_line = ""

                # Concatenate text with appropriate spacing
                for x_pos, text, font_size, is_bold in line_elements:
                    # Apply formatting based on properties
                    formatted_text = text

                    # Apply bold formatting if identified as bold text
                    if is_bold:
                        formatted_text = f"**{formatted_text}**"
                        is_heading = True

                    # Apply heading based on font size
                    if font_size > 12:  # Arbitrary threshold for headings
                        is_heading = True

                    # Add text with proper spacing
                    if text_line:
                        text_line += " " + formatted_text
                    else:
                        text_line = formatted_text

                # Apply heading formatting if detected
                if is_heading and len(text_line.strip()) < 100:  # Likely a heading
                    if font_size > 14:
                        markdown_content += f"### {text_line.strip()}\n\n"
                    else:
                        markdown_content += f"#### {text_line.strip()}\n\n"
                else:
                    # Add as regular paragraph if not empty
                    if text_line.strip():
                        markdown_content += text_line.strip() + "\n\n"

            # Extract tables for this page using tabula
            try:
                # Tabula uses 1-based page numbering
                tables = tabula.read_pdf(
                    pdf_path,
                    pages=str(page_num + 1),
                    multiple_tables=True,
                    lattice=True,
                    stream=True,
                    guess=True,
                )

                if tables and len(tables) > 0:
                    print(f"Found {len(tables)} tables on page {page_num + 1}")

                    for i, df in enumerate(tables):
                        if not df.empty and df.shape[0] > 0 and df.shape[1] > 0:
                            cleaned_df = clean_dataframe(df)
                            if not cleaned_df.empty:
                                # Add table header
                                markdown_content += f"#### Table {i+1}\n\n"

                                # Convert to markdown
                                try:
                                    table_md = cleaned_df.to_markdown(index=False)
                                    markdown_content += table_md + "\n\n"
                                except Exception as e:
                                    print(f"Error converting table to markdown: {e}")
                                    try:
                                        # Alternate method
                                        table_md = tabulate.tabulate(
                                            cleaned_df.values,
                                            headers=cleaned_df.columns,
                                            tablefmt="pipe",
                                        )
                                        markdown_content += table_md + "\n\n"
                                    except Exception as e2:
                                        print(
                                            f"Second attempt at table conversion failed: {e2}"
                                        )
            except Exception as e:
                print(f"Error extracting tables from page {page_num + 1}: {e}")

        # Post-process the markdown
        # Clean up excessive newlines and spacing
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        # Save to file if output path provided
        if output_path:
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(markdown_content)
            print(f"Extracted markdown with layout saved to {output_path}")

        return markdown_content

    except Exception as e:
        print(f"Error extracting PDF with layout: {e}")
        import traceback

        traceback.print_exc()
        return None


def process_layout_element(element, text_by_y):
    """
    Process a layout element and collect text with position information.

    Args:
        element: The layout element to process
        text_by_y: Dictionary to collect text by y-position
    """
    if isinstance(element, LTTextBox):
        for text_line in element:
            if isinstance(text_line, LTTextLine):
                # Get the y-position (rounded to handle slight variations)
                y_pos = round(text_line.y0)

                # Initialize list for this y-position if it doesn't exist
                if y_pos not in text_by_y:
                    text_by_y[y_pos] = []

                # Extract text content
                text = text_line.get_text().strip()

                # Extract font information if available
                font_size = 10  # Default size
                is_bold = False

                # Check contained character objects for font info
                for char in text_line:
                    if (
                        isinstance(char, LTChar)
                        and hasattr(char, "fontname")
                        and hasattr(char, "size")
                    ):
                        font_size = char.size
                        # Check if font name contains 'Bold'
                        if "bold" in char.fontname.lower():
                            is_bold = True
                        break

                # Store text with position and properties
                text_by_y[y_pos].append((text_line.x0, text, font_size, is_bold))

    elif isinstance(element, LTTextContainer):
        # Process text container
        text = element.get_text().strip()
        if text:
            # Get position
            y_pos = round(element.y0)

            # Initialize list for this y-position if it doesn't exist
            if y_pos not in text_by_y:
                text_by_y[y_pos] = []

            # Store text with position (default formatting)
            text_by_y[y_pos].append((element.x0, text, 10, False))

    elif isinstance(element, LTFigure):
        # Process contained elements recursively
        for child in element:
            process_layout_element(child, text_by_y)

    elif isinstance(element, LTImage):
        # Process image element
        if hasattr(element, "stream"):
            image_data = element.stream.get_data()
            if image_data:
                try:
                    # Create directory for images if it doesn't exist
                    image_dir = "extracted_images"
                    os.makedirs(image_dir, exist_ok=True)

                    # Generate unique filename based on image hash
                    image_hash = hash(image_data) & 0xFFFFFFFF
                    image_filename = f"{image_dir}/image_{image_hash}.png"

                    # Save the image
                    with open(image_filename, "wb") as f:
                        f.write(image_data)

                    # Get y-position for positioning in markdown content
                    y_pos = round(element.y0)

                    # Initialize list for this y-position if it doesn't exist
                    if y_pos not in text_by_y:
                        text_by_y[y_pos] = []

                    # Add image markdown reference to the text elements
                    image_md = f"![Image]({image_filename})"
                    text_by_y[y_pos].append((element.x0, image_md, 0, False))

                except Exception as e:
                    print(f"Error processing image: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract content from PDF files")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("-o", "--output", help="Path to save the extracted content")
    parser.add_argument(
        "--tables", action="store_true", help="Extract only tables in markdown format"
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Extract as plain text only (no markdown formatting)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Extract entire PDF as markdown with integrated tables",
    )
    parser.add_argument(
        "--layout",
        action="store_true",
        help="Extract PDF preserving layout information",
    )
    parser.add_argument(
        "--pages",
        default="all",
        help='Page numbers to extract from (e.g., "1,3,5-7" or "all")',
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="Extract images from PDF and include in markdown",
    )
    parser.add_argument(
        "--no-images",
        action="store_false",
        dest="images",
        help="Do not extract images from PDF",
    )
    parser.add_argument(
        "--image-dir",
        default="extracted_images",
        help="Directory to save extracted images",
    )

    # Set default for image extraction
    parser.set_defaults(images=True)

    args = parser.parse_args()

    # Default to markdown extraction if no specific format is requested
    if not args.tables and not args.markdown and not args.text and not args.layout:
        args.markdown = True

    # Handle image-only extraction
    if (
        args.images
        and not args.tables
        and not args.markdown
        and not args.text
        and not args.layout
    ):
        image_paths = extract_images_from_pdf(args.pdf_path, args.image_dir)
        print(f"Extracted {len(image_paths)} images to {args.image_dir}/")
        if len(image_paths) > 0:
            print("Extracted images:")
            for path in image_paths:
                print(f"  - {path}")
        return

    if args.layout:
        if args.output:
            extract_pdf_with_layout(
                args.pdf_path, args.output, args.pages, extract_images=args.images
            )
        else:
            markdown = extract_pdf_with_layout(
                args.pdf_path, pages=args.pages, extract_images=args.images
            )
            print("\nExtracted PDF as Markdown with Layout:")
            print("------------------------------------")
            print(markdown)
    elif args.markdown:
        if args.output:
            extract_pdf_as_markdown(
                args.pdf_path,
                args.output,
                args.pages,
                extract_images=args.images,
                image_dir=args.image_dir,
            )
        else:
            markdown = extract_pdf_as_markdown(
                args.pdf_path,
                pages=args.pages,
                extract_images=args.images,
                image_dir=args.image_dir,
            )
            print("\nExtracted PDF as Markdown:")
            print("-------------------------")
            print(markdown)
    elif args.tables:
        if args.output:
            extract_tables_as_markdown(args.pdf_path, args.output, args.pages)
        else:
            markdown = extract_tables_as_markdown(args.pdf_path, pages=args.pages)
            print("\nExtracted Tables (Markdown format):")
            print("-----------------------------------")
            print(markdown)
    elif args.text:
        if args.output:
            extract_text_from_pdf(args.pdf_path, args.output)
        else:
            text = extract_text_from_pdf(args.pdf_path)
            print("\nExtracted Text:")
            print("---------------")
            print(text)


if __name__ == "__main__":
    main()
