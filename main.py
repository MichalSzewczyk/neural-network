from preprocessing.data_loader import DataLoader

data_loader = DataLoader()
data = data_loader.load_file('project/dev.txt')
print(data)
