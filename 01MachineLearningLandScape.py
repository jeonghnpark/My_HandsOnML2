import sys
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import PyQt5

assert sys.version_info >= (3, 5)
import sklearn
import pandas as pd

assert sklearn.__version__ >= "0.20"
import urllib.request
import os


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


# data download


datapath = os.path.join("datasets", "lifesat","")

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/rickiepark/handson-ml2/master/"
os.makedirs(datapath, exist_ok=True)
url = DOWNLOAD_ROOT + "datasets/lifesat/" + "oecd_bil_2015.csv"
for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
    print("downloading..", filename)
    url = DOWNLOAD_ROOT + "datasets/lifesat/" + filename
    urllib.request.urlretrieve(url, datapath + filename)

# 적재

oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv", thousands=',', delimiter="\t", encoding="latin1",
                             na_values='n/a')

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
xx=country_stats['GDP per capita']
X = np.c_[country_stats['GDP per capita']]
y = np.c_[country_stats['Life satisfaction']]

country_stats.plot(kind='scatter', x="GDP per capita", y="Life satisfaction")
plt.show(block=False)
model = LinearRegression()
model.fit(X, y)

X_new = [[22587]]  # GDP for 키프로스
print("happyness for 키프로스", model.predict(X_new))

# ELA practice
path = os.path.join("datasets", "lifesat", "")  # datasets\lifesat\

oecd_bli = pd.read_csv(path + "oecd_bli_2015.csv", thousands=",")
oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
oecd_bli["Life satisfaction"].head()
gdp_per_capita = pd.read_csv(path + "gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1',
                             na_values='n/a')

gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
gdp_per_capita.set_index('Country', inplace=True)
full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
full_country_stats[["GDP per capita", 'Life satisfaction']].loc["United States"]
remove_indice = [0, 1, 6, 8, 33, 34, 35]
keep_indices = list(set(range(36)) - set(remove_indice))
# full_country_stats= full_country_stats[["GDP per capita", 'Life satisfaction']]
sample_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
sample_data = full_country_stats[["GDP per capita", 'Life satisfaction']]

missing_data=full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[remove_indice]

sample_data.plot(kind='scatter', x="GDP per capita", y="Life satisfaction",figsize=(5,3))
plt.axis([0,60000,0,10])
PROJECT_ROOT_DIR="."
CHAPTER_ID="fundamentals"
IMAGE_PATH=os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGE_PATH, exist_ok=True)

def save_fig(fig_id,tight_layout=True, fig_extension="png", resolution=300):
    path=os.path.join(IMAGE_PATH, fig_id+"."+fig_extension)
    print("saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)



position_text={"Hungary":(5000,1),
               "Korea":(18000,1.7),
               "France":(29000,2.4),
               "Australia":(40000,3.0),
               "United States":(52000,3.8)}

for country, pos_text in position_text.items():
    pos_data_x, pos_data_y=sample_data.loc[country]
    plt.annotate(country, xy=(pos_data_x, pos_data_y),xytext=pos_text,
                 arrowprops=dict(facecolor='black',width=0.5, shrink=0.1, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, 'ro')

plt.xlabel('GDP per capita')
save_fig('money_happy_scatterplot')
plt.show()
