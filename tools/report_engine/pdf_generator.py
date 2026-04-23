# src/pdf_generator.py

import os
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, CondPageBreak, Table, TableStyle
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas


class PageNumCanvas(canvas.Canvas):
    """Custom canvas to add page numbers to each page"""
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self.pages = []
        
    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()
        
    def save(self):
        page_count = len(self.pages)
        for page_num, page in enumerate(self.pages, 1):
            self.__dict__.update(page)
            self.draw_page_number(page_num, page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
        
    def draw_page_number(self, page_num, page_count):
        page_text = f"Page {page_num}"
        self.setFont("Helvetica", 9)
        self.drawRightString(
            A4[0] - 40,  # 40 points from right edge
            30,  # 30 points from bottom
            page_text
        )


class TOCDocTemplate(SimpleDocTemplate):
    """DocTemplate that collects TOC entries via afterFlowable notifications."""

    def afterFlowable(self, flowable):
        if hasattr(flowable, "toc_level"):
            level = flowable.toc_level
            text = flowable.toc_text
            key = getattr(flowable, "toc_key", None)
            if key:
                self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (level, text, self.page, key))


class PDFReportGenerator:

    def __init__(
        self,
        output_path="data/processed/drive_test_report.pdf",
        images_dir="data/images",
        processed_dir="data/processed"
    ):
        self.output_path = output_path
        self.images_dir = images_dir
        self.processed_dir = processed_dir

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.doc = TOCDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=50,
            bottomMargin=50
        )

        self.story = []
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    # --------------------------------------------------
    # STYLES
    # --------------------------------------------------
    def _setup_styles(self):
        self.styles.add(ParagraphStyle(
            name="CustomTitle",
            parent=self.styles["Heading1"],
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=30
        ))

        self.styles.add(ParagraphStyle(
            name="Section",
            parent=self.styles["Heading2"],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor("#1f4788")
        ))

        self.styles.add(ParagraphStyle(
            name="SubSection",
            parent=self.styles["Heading3"],
            fontSize=13,
            spaceBefore=12,
            spaceAfter=6
        ))

        self.styles.add(ParagraphStyle(
            name="Body",
            parent=self.styles["BodyText"],
            fontSize=11,
            leading=15
        ))
        
        # TOC Styles
        self.styles.add(ParagraphStyle(
            name="TOCHeading",
            parent=self.styles["Heading1"],
            fontSize=18,
            textColor=colors.HexColor("#1f4788"),
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name="TOCLevel1",
            parent=self.styles["Normal"],
            fontSize=11,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=3,
            spaceAfter=3,
            leading=14
        ))
        
        self.styles.add(ParagraphStyle(
            name="TOCLevel2",
            parent=self.styles["Normal"],
            fontSize=10,
            leftIndent=20,
            firstLineIndent=0,
            spaceBefore=2,
            spaceAfter=2,
            leading=12
        ))

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def add_text(self, title, text):
        if text and text.strip():
            self.story.append(Paragraph(title, self.styles["Section"]))
            self.story.append(Paragraph(text, self.styles["Body"]))
            self.story.append(Spacer(1, 12))

    def add_sub(self, title):
        self.story.append(Paragraph(title, self.styles["SubSection"]))

    def render_section(self, title: str, content: Any, level: int = 1):
        """
        Recursively render a section. `content` may be a string, dict or list.
        - string: render as a paragraph under `title`.
        - dict: if contains 'Overview' render it first; then iterate other keys
          and render each key as a subsection (dynamic, no hardcoding).
        - list: for list of dicts with 'name' use name as subsection title.
        """
        # choose style for title based on level
        if level == 1:
            title_style = self.styles["Section"]
        else:
            title_style = self.styles["SubSection"]

        # Skip rendering entirely if there's no content to avoid empty headings
        if content is None:
            return
        if isinstance(content, str) and not content.strip():
            return
        if isinstance(content, (dict, list)) and len(content) == 0:
            return

        # Render title
        if title:
            # Add bookmark/anchor for TOC
            if level == 1:
                title_para = Paragraph(f'<a name="{title}"/>{title}', title_style)
            else:
                title_para = Paragraph(f'<a name="{title}"/>{title}', title_style)
            self.story.append(title_para)

        # Render string content
        if isinstance(content, str):
            if content.strip():
                self.story.append(Paragraph(content, self.styles["Body"]))
                self.story.append(Spacer(1, 6))  # Reduced from 8 to 6
            return

        # Render dict content
        if isinstance(content, dict):
            # overview first
            overview = None
            for k in ("Overview", "overview"):
                if k in content and isinstance(content[k], str):
                    overview = content[k]
                    break
            if overview:
                self.story.append(Paragraph(overview, self.styles["Body"]))
                self.story.append(Spacer(1, 8))

            # Fallback: iterate keys dynamically, skip any overview/summary keys already rendered
            for k, v in content.items():
                if k in ("Overview", "overview"):
                    continue
                if k in ("Conclusion", "conclusion"):
                    # render at end
                    continue

                # If the value is string, render as subsection with that title
                if isinstance(v, str):
                    self.story.append(Paragraph(k, self.styles["SubSection"]))
                    self.story.append(Paragraph(v, self.styles["Body"]))
                    self.story.append(Spacer(1, 4))  # Reduced from 6 to 4
                else:
                    # nested structure: recurse
                    self.render_section(k, v, level=level + 1)

            # conclusion if present
            concl = content.get("Conclusion") or content.get("conclusion")
            if concl and isinstance(concl, str):
                self.story.append(Paragraph(concl, self.styles["Body"]))
                self.story.append(Spacer(1, 10))
            return

        # Render list content
        if isinstance(content, list):
            # If it's a list of dicts with 'name', render each as subsection
            rendered = False
            for item in content:
                if isinstance(item, dict) and item.get("name"):
                    name = item.get("name")
                    # derive a body text for the item
                    body = item.get("description") or item.get("text") or item.get("summary") or json.dumps(item)
                    self.story.append(Paragraph(name, self.styles["SubSection"]))
                    if isinstance(body, str) and body.strip():
                        self.story.append(Paragraph(body, self.styles["Body"]))
                    self.story.append(Spacer(1, 6))
                    rendered = True
                elif isinstance(item, str):
                    self.story.append(Paragraph(item, self.styles["Body"]))
                    rendered = True

            if not rendered:
                # fallback: render short joined representation
                joined = ", ".join(str(i) for i in content[:12])
                self.story.append(Paragraph(joined, self.styles["Body"]))
            self.story.append(Spacer(1, 8))
            return

        # fallback
        if content is not None:
            self.story.append(Paragraph(str(content), self.styles["Body"]))
            self.story.append(Spacer(1, 4))  # Reduced from 8

    def add_image(self, filename, subdir="kpi_maps", max_height=4*inch):
        path = os.path.join(self.images_dir, subdir, filename)
        if not os.path.exists(path):
            print(f" Warning: missing {path}")
            return

        img = Image(path)
        iw, ih = img.imageWidth, img.imageHeight
        scale = min((5.8*inch)/iw, max_height/ih, 1.0)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale

        self.story.append(img)
        self.story.append(Spacer(1, 6))  # Small spacing between images

    # --------------------------------------------------
    # COVER
    # --------------------------------------------------
    def add_cover(self, metadata):
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph("Drive Test Report", self.styles["CustomTitle"]))

        loc = metadata.get("location", {})
        self.story.append(Paragraph(
            f"{loc.get('city','')} , {loc.get('country','')}",
            self.styles["Body"]
        ))

        self.story.append(Spacer(1, 12))
        self.story.append(Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y')}",
            self.styles["Body"]
        ))

        self.story.append(PageBreak())
        
    def add_table_of_contents(self):
        """Add professional TOC with dotted lines and dynamic page numbers"""
        # Title
        self.story.append(Paragraph("<b>Table of Contents</b>", self.styles["TOCHeading"]))
        self.story.append(Spacer(1, 0.2 * inch))
        
        # Create TOC object with proper formatting
        from reportlab.platypus.tableofcontents import TableOfContents
        toc = TableOfContents()
        
        # Use existing TOC styles from _setup_styles
        toc.levelStyles = [
            self.styles['TOCLevel1'],
            self.styles['TOCLevel2'],
        ]
        
        self.story.append(toc)
        self.story.append(PageBreak())

    def add_toc_heading(self, text, style, level, key, toc_text=None):
        """Add a heading paragraph and register it for TOC via afterFlowable."""
        para = Paragraph(text, style)
        para.toc_level = level
        para.toc_text = toc_text or text
        para.toc_key = key
        self.story.append(para)

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------
    def generate_report(self, report_text, metadata, verbose=False):

        self.add_cover(metadata)
        self.add_table_of_contents()

        # 1. Introduction
        self.add_toc_heading('1. Introduction', self.styles["Section"], 0, "sec1")
        if report_text.get("Introduction"):
            self.story.append(Paragraph(report_text["Introduction"], self.styles["Body"]))
        
        # 2. Area Summary
        self.story.append(CondPageBreak(3 * inch))
        self.add_toc_heading('2. Area Summary', self.styles["Section"], 0, "sec2")
        self.render_section(None, report_text.get("Area Summary", ""))
        self.add_image("base_route_map.png")
        
        # 3. Drive Summary
        self.story.append(PageBreak())
        self.add_toc_heading('3. Drive Summary', self.styles["Section"], 0, "sec3")
        if report_text.get("Drive Summary"):
            self.story.append(Paragraph(report_text["Drive Summary"], self.styles["Body"]))
        self.add_image("drive_summary.png", subdir="kpi_analysis")

        # 4. KPI Summary
        self.story.append(CondPageBreak(3 * inch))
        self.add_toc_heading('4. KPI Summary', self.styles["Section"], 0, "sec4")
        kpi_text = report_text.get("KPI Summary")
        if not kpi_text or not isinstance(kpi_text, str) or not kpi_text.strip():
            kpi_text = (
                "A concise executive summary of overall KPI performance across the drive, "
                "highlighting overall network health and any major issues observed."
            )
        self.story.append(Paragraph(kpi_text, self.styles["Body"]))
        self.add_image("kpi_summary.png", subdir="kpi_analysis")
        self.add_image("session_table.png", subdir="kpi_analysis")

        # 5. MAP VIEW
        self.story.append(PageBreak())
        self.add_toc_heading('5. Map View', self.styles["Section"], 0, "sec5")

        def kpi_sub(label, anchor, toc_label, text, imgs):
            self.add_toc_heading(label, self.styles["SubSection"], 1, anchor, toc_label)
            if text:
                self.story.append(Paragraph(text, self.styles["Body"]))
                self.story.append(Spacer(1, 6))
            for img in imgs:
                self.add_image(**img)

        # a) Band
        kpi_sub("a) Band", "sec5a", "a) Band", report_text.get("Map View - Band", ""), [
            {"filename": "band_map.png"},
            {"filename": "band_pie.png", "subdir": "kpi_analysis"},
            {"filename": "band_table.png", "subdir": "kpi_analysis"},
        ])

        # b) RSRP
        kpi_sub("b) RSRP", "sec5b", "b) RSRP", report_text.get("Map View - RSRP", ""), [
            {"filename": "rsrp_map.png"},
            {"filename": "rsrp_poor_regions.png"},
            {"filename": "rsrp_range_table.png", "subdir": "kpi_analysis"},
            {"filename": "cdf_rsrp.png", "subdir": "kpi_analysis"},
        ])

        # c) RSRQ
        kpi_sub("c) RSRQ", "sec5c", "c) RSRQ", report_text.get("Map View - RSRQ", ""), [
            {"filename": "rsrq_map.png"},
            {"filename": "rsrq_poor_regions.png"},
            {"filename": "rsrq_range_table.png", "subdir": "kpi_analysis"},
            {"filename": "cdf_rsrq.png", "subdir": "kpi_analysis"},
        ])

        # d) SINR
        kpi_sub("d) SINR", "sec5d", "d) SINR", report_text.get("Map View - SINR", ""), [
            {"filename": "sinr_map.png"},
            {"filename": "sinr_range_table.png", "subdir": "kpi_analysis"},
            {"filename": "cdf_sinr.png", "subdir": "kpi_analysis"},
        ])

        # e) DL Throughput
        kpi_sub("e) DL Throughput", "sec5e", "e) DL Throughput", report_text.get("Map View - DL Throughput", ""), [
            {"filename": "dl_map.png"},
            {"filename": "dl_range_table.png", "subdir": "kpi_analysis"},
            {"filename": "cdf_dl_tpt.png", "subdir": "kpi_analysis"},
        ])

        # f) UL Throughput
        kpi_sub("f) UL Throughput", "sec5f", "f) UL Throughput", report_text.get("Map View - UL Throughput", ""), [
            {"filename": "ul_map.png"},
            {"filename": "ul_range_table.png", "subdir": "kpi_analysis"},
            {"filename": "cdf_ul_tpt.png", "subdir": "kpi_analysis"},
        ])

        # g) MOS
        kpi_sub("g) MOS", "sec5g", "g) MOS", report_text.get("Map View - MOS", ""), [
            {"filename": "mos_map.png"},
            {"filename": "mos_range_table.png", "subdir": "kpi_analysis"},
            {"filename": "cdf_mos.png", "subdir": "kpi_analysis"},
        ])

        # 6. PCI Summary
        self.story.append(PageBreak())
        self.add_toc_heading('6. PCI Summary', self.styles["Section"], 0, "sec6")
        if report_text.get("PCI Summary"):
            self.story.append(Paragraph(report_text["PCI Summary"], self.styles["Body"]))

        self.add_image("pci_map.png")
        self.add_image("pci_distribution.png", subdir="kpi_analysis")
        self.add_image("cdf_pci.png", subdir="kpi_analysis")

        self.add_toc_heading('a) Top 30 PCI Values', self.styles["SubSection"], 1, "sec6a")
        self.add_image("pci_table.png", subdir="kpi_analysis")

        self.add_toc_heading('b) PCI with Poor RSRP', self.styles["SubSection"], 1, "sec6b")
        self.add_image("pci_poor_rsrp.png", subdir="kpi_analysis")

        self.add_toc_heading('c) PCI with Poor RSRQ', self.styles["SubSection"], 1, "sec6c")
        self.add_image("pci_poor_rsrq.png", subdir="kpi_analysis")

        # 7. App Analytics
        self.story.append(CondPageBreak(3 * inch))
        self.add_toc_heading('7. App Analytics', self.styles["Section"], 0, "sec7")
        self.add_image("app_analytics_part1.png", subdir="kpi_analysis")
        self.add_image("app_analytics_part2.png", subdir="kpi_analysis")

        # 8. Indoor/Outdoor Summary
        self.story.append(CondPageBreak(3 * inch))
        self.add_toc_heading('8. Indoor/Outdoor Summary', self.styles["Section"], 0, "sec8")
        self.add_image("indoor_outdoor_stats.png", subdir="kpi_analysis")

        # 9. Performance Summary
        self.story.append(CondPageBreak(3 * inch))
        self.add_toc_heading('9. Performance Summary', self.styles["Section"], 0, "sec9")

        self.add_toc_heading('a) Network Quality Metrics', self.styles["SubSection"], 1, "sec9a")
        self.add_image("network_quality_summary.png", subdir="kpi_analysis")

        self.add_toc_heading('b) Speed Metrics', self.styles["SubSection"], 1, "sec9b")
        self.add_image("speed_hist.png", subdir="kpi_analysis")

        self.add_toc_heading('c) Latency Distribution', self.styles["SubSection"], 1, "sec9c")
        self.add_image("latency_hist.png", subdir="kpi_analysis")

        self.add_toc_heading('d) Jitter Distribution', self.styles["SubSection"], 1, "sec9d")
        self.add_image("jitter_hist.png", subdir="kpi_analysis")

        # 10. Handover Analysis
        self.story.append(PageBreak())
        self.add_toc_heading('10. Handover Analysis', self.styles["Section"], 0, "sec10")
        self.add_image("handover_map.png")

        # Build with multiple passes for TOC and page numbers
        self.doc.multiBuild(self.story, canvasmaker=PageNumCanvas)


def generate_pdf_report(
    metadata_path="data/processed/report_metadata.json",
    report_text_path="data/processed/report_text.json",
    output_path="data/processed/drive_test_report.pdf",
    images_dir="data/images",
    verbose=False
):
    with open(metadata_path) as f:
        metadata = json.load(f)
    with open(report_text_path) as f:
        report_text = json.load(f)

    gen = PDFReportGenerator(output_path, images_dir)
    gen.generate_report(report_text, metadata, verbose)
    return output_path


def main():
    generate_pdf_report(verbose=True)


if __name__ == "__main__":
    main()

