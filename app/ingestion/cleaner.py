import re


class TextCleaner:
    """
    Cleans and normalizes raw text from technical documentation while preserving 
    code blocks, inline code snippets, structural indentations, and syntax.
    """

    def __init__(self):
        # Match code blocks delimited by triple backticks (```code```)
        self.code_block_pattern = re.compile(r"(```[\s\S]*?```)")
        # Match HTML tags that are not code snippets
        self.html_tag_pattern = re.compile(r"<[^>]+>")
        # Match duplicate non-newline whitespace (spaces/tabs)
        self.horizontal_space_pattern = re.compile(r"[ \t]+")
        # Match 3 or more consecutive newlines
        self.excessive_newlines_pattern = re.compile(r"\n{3,}")

    def clean(self, text: str) -> str:
        """
        Cleans input document text safely.

        Args:
            text (str): Raw string content from document loader.

        Returns:
            str: Cleaned and normalized text content.
        """
        if not text or not text.strip():
            return ""

        # Step 1: Split text into code blocks and prose blocks to protect code syntax
        tokens = self.code_block_pattern.split(text)

        cleaned_tokens = []
        for token in tokens:
            # If token is a fenced code block, keep it intact except trimming trailing space
            if token.startswith("```") and token.endswith("```"):
                cleaned_tokens.append(token.strip())
            else:
                # Clean prose text safely
                cleaned_text = self._clean_prose(token)
                if cleaned_text:
                    cleaned_tokens.append(cleaned_text)

        return "\n\n".join(cleaned_tokens).strip()

    def _clean_prose(self, prose: str) -> str:
        """Applies normalization rules to non-code prose text."""
        # Standardize carriage returns
        prose = prose.replace("\r\n", "\n").replace("\r", "\n")

        # Remove raw HTML tags (if any exist from web scrapes/conversion)
        prose = self.html_tag_pattern.sub("", prose)

        # Replace non-breaking spaces with standard space
        prose = prose.replace("\xa0", " ")

        # Collapse horizontal spaces and tabs into a single space
        prose = self.horizontal_space_pattern.sub(" ", prose)

        # Collapse 3+ consecutive newlines down to 2 newlines (paragraph boundary)
        prose = self.excessive_newlines_pattern.sub("\n\n", prose)

        # Strip spaces at the start/end of individual lines
        lines = [line.strip() for line in prose.split("\n")]
        
        return "\n".join(lines).strip()