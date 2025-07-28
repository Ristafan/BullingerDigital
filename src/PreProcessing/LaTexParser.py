

class LaTexParser:
    def __init__(self):
        self.hbbw_number = None
        self.from_pers_id = None
        self.to_pers_id = None
        self.place_id = None
        self.date = None
        self.document_type = None

        self.regest = []
        self.text = []

    def parse_latex(self, filepath):
        """
        Parses a LaTeX document and extracts sections, subsections, and their content.
        :param filepath: Path to the LaTeX file.
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Extract document type from the first ten lines
        start_line_briefkopf = 0
        end_line_briefkopf = 10

        for line in lines:
            if "\\begin{HBBWBriefkopf}" in line:
                start_line_briefkopf = lines.index(line)
                break
            if "\\end{HBBWBriefkopf}" in line:
                end_line_briefkopf = lines.index(line)
                break

        self.parse_hbbw_briefkopf(lines[start_line_briefkopf:end_line_briefkopf])

        # Check if the document type is at the end of the file
        self.get_document_type(lines[-3:])

        # Extracting the regest and text sections


    def parse_hbbw_briefkopf(self, lines):
        # Extracting the HBBW number from the first line
        index_number_starting = lines[1].find("{") + 1
        self.hbbw_number = int(lines[1][index_number_starting:-2])

        # Extracting the sender and recipient IDs
        index_from_starting = lines[3].find("\"") + 1
        self.from_pers_id = lines[3][index_from_starting:-4]
        index_to_starting = lines[4].find("\"") + 1
        self.to_pers_id = lines[4][index_to_starting:-4]

        # Extracting the place ID
        index_place_starting = lines[5].find("\"") + 1
        self.place_id = lines[5][index_place_starting:-4]

        # Extracting the date
        index_date_starting = lines[6].find("\"") + 1
        self.date = lines[6][index_date_starting:-4]

    def get_document_type(self, lines):
        for line in lines:
            if line.startswith("%"):
                self.document_type = line[3:-1]

    def latex_cmds_coordinator(self, line):
        # Listing of different LaTeX commands and how to handle them
        commands = {
            "\\Korrespondenten": None,
            "\\HBBWOrtDatum": None,
            "\\HBBWSignatur": None,
            "\\HBBWDruck": None
        }



if __name__ == "__main__":
    path = "G:/Meine Ablage/ILU/HBBW21_tex/002-18335-Bullinger_an_Johannes_Stumpf,[Zürich],_3._Januar_1548.tex"  # Replace with your LaTeX file path

    parser = LaTexParser()
    parser.parse_latex(path)
    print(f"HBBW Number: {parser.hbbw_number}")
    print(f"From Person ID: {parser.from_pers_id}")
    print(f"To Person ID: {parser.to_pers_id}")
    print(f"Place ID: {parser.place_id}")
    print(f"Date: {parser.date}")
    print(f"Document Type: {parser.document_type}")
