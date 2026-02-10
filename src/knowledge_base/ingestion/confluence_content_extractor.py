from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import quote_plus
import logging

# Example usage imports
from constants import UWF_CONFLUENCE_BASE_URL, UWF_CONFLUENCE_PAGE_SPACE

logger = logging.getLogger(__name__)


class ConfluenceContentExtractor:
    """
    Parses raw Confluence HTML, sanitizes custom tags, and converts to Markdown.
    """

    def extract(
        self,
        raw_html: str,
        base_url: str = UWF_CONFLUENCE_BASE_URL,
        space_key: str = UWF_CONFLUENCE_PAGE_SPACE,
    ) -> str:
        """
        Takes raw HTML string and returns formatted Markdown with absolute links.

        Args:
            raw_html (str): The storage format HTML.
            base_url (str): The instance URL (e.g., https://confluence.uwf.edu).
            space_key (str): The space key of the current page (used as default for internal links).

        Returns:
            str: Cleaned Markdown string.
        """
        if not raw_html:
            return ""

        # Use 'lxml' for speed and better handling of XML tags like <ac:link>
        soup = BeautifulSoup(raw_html, "lxml")
        print(soup)

        # Remove Confluence macro noise
        # Remove all <ac:parameter> tags
        # These contain the unnecessary text "BLOCK", "true", "20px", etc.
        for param in soup.find_all("ac:parameter"):
            param.decompose()

        # Remove images entirely (needed for text embeddings)
        for img in soup.find_all("ac:image"):
            img.decompose()

        # Remove Table of Contents
        for toc in soup.find_all("ac:structured-macro", attrs={"ac:name": "toc"}):
            toc.decompose()

        # Unwrap layouts to free the text
        for tag in soup.find_all(["ac:layout", "ac:layout-section", "ac:layout-cell", "ac:rich-text-body"]):
            tag.unwrap()

        # Handle Confluence internal links (<ac:link>)
        for link in soup.find_all("ac:link"):
            page_ref = link.find("ri:page")

            if page_ref and page_ref.has_attr("ri:content-title"):
                page_title = page_ref["ri:content-title"]

                link_text = page_title

                # Check for link text (CDATA), otherwise default to page title
                # 'lxml' parser will ignore <![CDATA[...]]> block, meaning .get_text() will return an emtpy string
                link_body = link.find("ac:plain-text-link-body")
                if link_body:
                    candidate_text = link_body.get_text(
                        strip=True
                    )  # Trims whitespace and returns empty string if tag contains only whitespace
                    if candidate_text:  # ignores empty strings
                        link_text = candidate_text

                # Determine the Space (default to current space if missing)
                target_space = page_ref.get("ri:space-key", space_key)

                # Build the Absolute URL
                safe_title = quote_plus(str(page_title))
                full_url = f"{base_url}/display/{target_space}/{safe_title}"

                # Replace XML with standard HTML <a> tag
                new_tag = soup.new_tag("a", href=full_url)
                new_tag.string = str(link_text)

                link.replace_with(new_tag)

        # Clean up empty paragraphs
        for p in soup.find_all("p"):
            if not p.get_text(strip=True):
                p.decompose()

        # Convert to Markdown
        # 'body' logic: lxml adds <html><body> tags automatically.
        # We process just the body content to avoid extra noise.
        body_content = soup.body if soup.body else soup
        clean_html = str(body_content)
        print(clean_html)

        # Markdownify strips tags but keeps text
        # ATX ensures headers use the (#) format
        return md(clean_html, heading_style="ATX").strip()


if __name__ == "__main__":  # pragma: no cover
    raw_html = """<h1>Overview</h1><p>This page contains information about academic advising.\xa0</p>
    <h1>Pages on this Topic</h1><p><br /></p><p><ac:link><ri:page ri:content-title="Advising Syllabus" /><ac:plain-text-link-body><![CDATA[Advising Syllabus]]></ac:plain-text-link-body></ac:link></p>
    <ul><li>What is Academic Advising</li><li>Student\'s Role &amp; Responsibilities</li><li>Advisor\'s Role &amp; Responsibilities</li></ul>
    <p><a href="https://confluence.uwf.edu/display/public/Identifying+your+advisor">How to locate your academic advising and student support team members</a>\xa0</p>
    <p><a href="https://confluence.uwf.edu/display/public/Schedule+an+Appointment+with+your+Academic+Advisor+or+Student+Support+staff">How to schedule an academic advising or student support appointment</a>\xa0</p>
    <p><ac:link><ri:page ri:content-title="How to cancel or reschedule an academic advising or student support appointment" /></ac:link></p>
    <p>How to prepare for your appointment</p><p><br /></p>"""

    raw_html = """<h1>Overview</h1><p>This page contains information about academic advising.\xa0</p>
    <h1>Pages on this Topic</h1><p><br /></p>"""

    raw_html = "<p>Raw HTML content...</p>"

    # raw_html = '<ac:layout><ac:layout-section ac:type="two_right_sidebar"><ac:layout-cell><h1>Overview</h1><ac:structured-macro ac:name="excerpt" ac:schema-version="1" ac:macro-id="a4c22ac9-fd74-4624-a27b-c3560a82cd6c"><ac:parameter ac:name="atlassian-macro-output-type">BLOCK</ac:parameter><ac:rich-text-body>LinkedIn Learning is<span style="letter-spacing: 0.0px;">\xa0</span>an on-demand library of instructional videos covering the latest business, technology and creative skills<span style="letter-spacing: 0.0px;">. Faculty, staff, and students have access to LinkedIn Learning through MyUWF. </span>This page provides the instructions for accessing LinkedIn Learning.\xa0</ac:rich-text-body></ac:structured-macro><h1>Instructions</h1><p>Below are the step by step instructions to access LinkedIn Learning.\xa0</p><h3>Step 1</h3><p>Login to <a href="https://my.uwf.edu/">My.UWF.edu</a>.\xa0</p><h3>Step 2</h3><p>Search for LinkedIn Learning.</p><p><ac:image ac:border="true" ac:title="MyUWF Homepage" ac:alt="MyUWF Homepage with arrow pointing to the LinkedIn Learning result in the search box. " ac:height="250" ac:width="518"><ri:attachment ri:filename="2022-08-10_15-44-00.jpg" /></ac:image></p><h3>Step 3</h3><p>Then click the <strong>Sounds good</strong> button to continue.\xa0\xa0</p><p><ac:image ac:border="true" ac:title="Welcome to LinkedIn Learning." ac:alt="This is a screenshot of the Welcome to LinkedIn Learning page. There is a Sounds good button on the page if you are ready to proceed. " ac:height="250" ac:width="452"><ri:attachment ri:filename="2022-08-10_15-52-08.jpg" /></ac:image></p><h3>Step 4</h3><p>Choose topics that interest you if you like, if not, click Continue.\xa0</p><p><ac:image ac:border="true" ac:title="Topics of Interests" ac:alt="Topics of Interests page in LinkedIn Learning. Boxes containing topics of interest.  " ac:height="250" ac:width="457"><ri:attachment ri:filename="2022-08-10_15-46-49.jpg" /></ac:image></p><h3>Step 5</h3><p>If you like, you can set a weekly goal. This is up to you. You can choose to <strong>Set a Goal</strong> or click <strong>Maybe later</strong>.\xa0</p><p><ac:image ac:border="true" ac:title="Weekly Goal Setting" ac:alt="This is an image of the Weekly Goal Setting page. The options are 15, 30, 60, and 120 minutes per week. There is a section on the page that recommends that users keep it casual and start with half of an hour. There is a box to choose Maybe Later, and a box to choose Set Goal. " ac:height="250" ac:width="454"><ri:attachment ri:filename="2022-08-10_15-47-34.jpg" /></ac:image></p><h3>Step 6</h3><p>Here you can connect your LinkedIn profile if you like. If you do not want to connect your profile, click the <strong>Not now</strong> button. If you click <strong>Not now</strong>, you will be taken to <strong>Step 7</strong> below. If you choose <strong>Connect LinkedIn account</strong>, your screen will look like <strong>Step 9</strong> further down on this page. You can always go back and connect your LinkedIn account later if you like.\xa0</p><p><ac:image ac:border="true" ac:title="Connect your LinkedIn Account" ac:alt="LinkedIn Profile Creation" ac:height="250" ac:width="466"><ri:attachment ri:filename="2022-08-10_15-47-53.jpg" /></ac:image></p><h3>Step 7</h3><p>Click the <strong>Start learning</strong> button.\xa0</p><p><ac:image ac:border="true" ac:title="Start Learning" ac:alt="Start Learning" ac:height="250" ac:width="455"><ri:attachment ri:filename="2022-08-10_16-38-37.jpg" /></ac:image></p><h3>Step 8</h3><p>Below is a screenshot of the LinkedIn Learning homepage. On your homepage you can search for topics in the Search bar. You are also provided Top Picks based on the topics that you are interested in. You can also go back and connect your LinkedIn Profile.\xa0</p><p><ac:image ac:border="true" ac:title="LinkedIn user homepage" ac:alt="This is a screenshot of the LinkedIn user homepage. The homepage has a sections to search for topics, a section of Top picks for you, and if you did not connected your LinkedIn profile, there is an option to Finish Up or choose Not now. " ac:height="250" ac:width="453"><ri:attachment ri:filename="2022-08-10_15-48-59.jpg" /></ac:image></p><h3>Step 9</h3><p>If you would like to continue to connect your LinkedIn Profile to your LinkedIn Learning account, click the <strong>Connect my LinkedIn Account</strong> button.\xa0</p><p><ac:image ac:border="true" ac:title="Confirm Identity to Connect LinkedIn Account" ac:alt="This is a screenshot of the Connect your LinkedIn account page. There is a Connect my LinkedIn Account box.  page. If you clicked on Connect your Lin on this page. " ac:height="250" ac:width="461"><ri:attachment ri:filename="2022-08-10_15-49-29.jpg" /></ac:image></p><h3>Step 10</h3><p>Here you can <strong>Learn more</strong>, <strong>Accept Terms and Continue</strong>, or review <strong>How it works</strong>. This step is up to you.\xa0</p><p><ac:image ac:border="true" ac:title="Confirm Identity to Connect LinkedIn Account" ac:alt="This is a screenshot of the Confirm Identity to Connect LinkedIn Account page. If you clicked on Connect your LinkedIn account, you can Accept and Continue on this page. " ac:height="250" ac:width="507"><ri:attachment ri:filename="2022-08-10_15-49-48.jpg" /></ac:image></p><h3>Step 11</h3><p>Once you connect your LinkedIn account you will be taken to your LinkedIn Learning homepage. You can search for topics or explore the Top Picks.\xa0</p><p><ac:image ac:border="true" ac:title="LinkedIn Learning Homepage" ac:alt="This is a screenshot of the LinkedIn Learning Homepage." ac:height="250" ac:width="453"><ri:attachment ri:filename="2022-08-10_15-50-27.jpg" /></ac:image></p><p><br /></p></ac:layout-cell><ac:layout-cell><p class="auto-cursor-target"><br /></p><ac:structured-macro ac:name="panel" ac:schema-version="1" ac:macro-id="29c220c1-ae09-497d-8b3b-63028fc528c0"><ac:parameter ac:name="title">On this page</ac:parameter><ac:rich-text-body><p><ac:structured-macro ac:name="toc" ac:schema-version="1" ac:macro-id="5189f856-247e-4ce4-bdf7-cfddc66e185f"><ac:parameter ac:name="outline">true</ac:parameter><ac:parameter ac:name="indent">20px</ac:parameter><ac:parameter ac:name="style">none</ac:parameter><ac:parameter ac:name="printable">false</ac:parameter></ac:structured-macro></p></ac:rich-text-body></ac:structured-macro><p><br /></p></ac:layout-cell></ac:layout-section></ac:layout>'

    # raw_html = """
    #         <p>
    #             <ac:link>
    #                 <ri:page ri:content-title="Fallback Title" />
    #                 <ac:plain-text-link-body><![CDATA[]]></ac:plain-text-link-body>
    #             </ac:link>

    #             <ac:link>
    #                 <ri:page ri:content-title="Policies" ri:space-key="HR" />
    #                 <ac:plain-text-link-body>HR Policies</ac:plain-text-link-body>
    #             </ac:link>

    #             <ac:link>
    #                 <ri:page ri:content-title="Q&A: What is it?" />
    #                 <ac:plain-text-link-body>Click Here</ac:plain-text-link-body>
    #             </ac:link>
    #         </p>
    #         """
    extractor = ConfluenceContentExtractor()

    md_string = extractor.extract(raw_html, base_url=UWF_CONFLUENCE_BASE_URL, space_key=UWF_CONFLUENCE_PAGE_SPACE)

    print(md_string)
