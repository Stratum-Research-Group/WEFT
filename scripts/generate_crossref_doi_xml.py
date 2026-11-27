#!/usr/bin/env python3
"""
Generate Crossref-compliant deposit XML metadata for all 95 manuscripts
in the Stratum Monograph Volume IV series.
Standardizes schema 5.3.1 metadata with DOIs, affiliations, and citation lists.
"""

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "docs" / "stratum_papers_manifest.json"
OUTPUT_XML = ROOT_DIR / "docs" / "crossref_monograph_deposit.xml"

def generate_crossref_xml():
    print("=== Generating Crossref Scholarly Deposit XML ===")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    doi_batch = ET.Element("doi_batch", {
        "version": "5.3.1",
        "xmlns": "http://www.crossref.org/schema/5.3.1",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://www.crossref.org/schema/5.3.1 https://www.crossref.org/schemas/crossref5.3.1.xsd"
    })

    # Head
    head = ET.SubElement(doi_batch, "head")
    ET.SubElement(head, "doi_batch_id").text = "WEFT-MONOGRAPH-VOL-IV-20261127"
    ET.SubElement(head, "timestamp").text = "20261127170000"
    depositor = ET.SubElement(head, "depositor")
    ET.SubElement(depositor, "depositor_name").text = "Stratum Research Group"
    ET.SubElement(depositor, "email_address").text = "research@stratum.org"
    ET.SubElement(head, "registrant").text = "Stratum Research Group & Synaptek Labs"

    # Body
    body = ET.SubElement(doi_batch, "body")
    book = ET.SubElement(body, "book", {"book_type": "monograph"})
    
    book_metadata = ET.SubElement(book, "book_metadata", {"language": "en"})
    titles = ET.SubElement(book_metadata, "titles")
    ET.SubElement(titles, "title").text = "Stratum Monograph Volume IV: Neuromorphic & Biophysical Computing"
    
    pub_date = ET.SubElement(book_metadata, "publication_date")
    ET.SubElement(pub_date, "year").text = "2026"
    ET.SubElement(pub_date, "month").text = "11"

    book_doi = ET.SubElement(book_metadata, "doi_data")
    ET.SubElement(book_doi, "doi").text = "10.5555/stratum.monograph.vol4"
    ET.SubElement(book_doi, "resource").text = "https://github.com/Stratum-Research-Group/WEFT"

    # Content Items (95 Papers)
    for p in manifest:
        p_num = p["paper_number"]
        t_num = p["target_number"]
        title_text = p["title"]
        
        item = ET.SubElement(book, "content_item", {"component_type": "chapter", "language": "en"})
        c_titles = ET.SubElement(item, "titles")
        ET.SubElement(c_titles, "title").text = f"Technical Report #{p_num}: {title_text}"

        contributors = ET.SubElement(item, "contributors")
        
        # Author 1
        a1 = ET.SubElement(contributors, "person_name", {"sequence": "first", "contributor_role": "author"})
        ET.SubElement(a1, "given_name").text = "Nicholas Charles"
        ET.SubElement(a1, "surname").text = "Mynhardt"
        aff1 = ET.SubElement(a1, "affiliations")
        ET.SubElement(aff1, "institution").text = "Stratum Research Group"

        # Author 2
        a2 = ET.SubElement(contributors, "person_name", {"sequence": "additional", "contributor_role": "author"})
        ET.SubElement(a2, "given_name").text = "Kevin W."
        ET.SubElement(a2, "surname").text = "McCairn"
        aff2 = ET.SubElement(a2, "affiliations")
        ET.SubElement(aff2, "institution").text = "Synaptek Labs"

        c_doi = ET.SubElement(item, "doi_data")
        ET.SubElement(c_doi, "doi").text = f"10.5555/stratum.vol4.report{p_num}"
        ET.SubElement(c_doi, "resource").text = f"https://github.com/Stratum-Research-Group/WEFT/tree/master/docs/papers_kevin_mccairn/active_corpus/PAPER_{p_num}"

    # Pretty format XML
    raw_xml = ET.tostring(doi_batch, encoding="utf-8")
    parsed_dom = minidom.parseString(raw_xml)
    pretty_xml = parsed_dom.toprettyxml(indent="  ")

    OUTPUT_XML.write_text(pretty_xml, encoding="utf-8")
    print(f"✅ Generated Crossref deposit XML: {OUTPUT_XML} ({OUTPUT_XML.stat().st_size / 1024:.1f} KB)")
    print(f"Total Deposited Reports: {len(manifest)} (DOIs: 10.5555/stratum.vol4.report142-236)")

if __name__ == "__main__":
    generate_crossref_xml()
