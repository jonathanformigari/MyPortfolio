import pandas

class excelhandling:
    
    def __init__(self, file_path, file_name, sheet_list = None):
        self.file_path = file_path
        self.file_name = file_name
        self.sheet_list = sheet_list
        self.data_frame = pandas.read_excel(self.file_path + self.file_name, sheet_list)
    
    def get_file_name(self):
        return self.file_name

    def get_file_path(self):
        return self.file_path

    def get_dataframe(self):
        return self.data_frame.copy()

    #def clean_header_lines():



excel_file = excelhandling("C:\\Users\\Jonathan\\Documents\\Lernen\\", "Teste_Abrir_Excel.xlsx")

# excel_file = excelhandling("C:Users\Jonathan\Documents\Lernen", "Teste_Abrir_Excel.xlsx")

print(excel_file.get_dataframe())