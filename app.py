from main import DatasetAnalyzer
import pandas as pd


if __name__ == "__main__":
    dataset_path = "Data/Luxury_Products_Apparel_Data.csv"
    data = pd.read_csv(dataset_path)
    analyzer = DatasetAnalyzer(data)
    analyzer.user_interface()
