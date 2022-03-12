import pandas as pd
import matplotlib.pyplot as plt

#pandas data table representation
def pandatut01():
    df = pd.DataFrame(
        {
            "Name": [
                "Braund, Mr. Owen Harris",
                "Allen, Mr. William Henry",
                "Bonnell, Miss. Elizabeth",
            ],
            "Age": [22, 35, 58],
            "Gender": ["male", "male", "female"],
        })
    print(df)
    print(df["Age"])
    print(df["Gender"])
    print(df["Name"])

    #creating series
    ages = pd.Series([22, 35, 58], name="Age")
    print(ages)

    print(df["Age"].max())
    print(ages.min())
    print(df.describe())

def pandatut02():
    titanic = pd.read_csv("data/titanic.csv")
    print(titanic)
    print("describe() function")
    print(titanic.describe())
    print("head() function")
    print(titanic.head(10))
    print("tail() function")
    print(titanic.tail(10))
    print(titanic.dtypes)
    print(titanic.info())

def pandatut03():
    titanic = pd.read_csv("data/titanic.csv")
    ages = titanic["Age"]
    print(ages)
    print(ages.head(10))
    print(type(titanic["Age"]))
    print(titanic["Age"].shape)

    age_gender = titanic[["Age", "Sex"]]
    print(age_gender)
    print(age_gender.head())
    print(type(age_gender))
    print(ages[6:10])

    #How do I filter specific rows from a DataFrame?
    above_35 = titanic[titanic["Age"] > 35]
    print(above_35)
    print(titanic["Age"] > 35)
    print(above_35.shape)

    class_23 = titanic[titanic["Pclass"].isin([2, 3])]
    print(class_23.head())
    class_23 = titanic[(titanic["Pclass"] == 2) | (titanic["Pclass"] == 3)]
    print(class_23.head())
    age_no_na = titanic[titanic["Age"].notna()]
    print(age_no_na.head())

    #How do I select specific rows and columns from a DataFrame?

    adult_names = titanic.loc[titanic["Age"] > 35,  "Name"]
    print(adult_names.head())

    print(titanic.iloc[9:25, 2:5])
    titanic.iloc[:, 3] = "anonymous"
    print(titanic.head())

    print(titanic["Age"].mean())
    print(titanic[["Age", "Fare"]].median())
    print(titanic[["Age", "Fare"]].describe())

def pandatut04():
    air_quality = pd.read_csv("data/air_quality_no2.csv", index_col=0, parse_dates=True)
    print(air_quality)
    print(air_quality.head())
    print(air_quality.tail())
    air_quality.plot()
    plt.show()

    air_quality["station_paris"].plot()
    plt.show()

    air_quality.plot.scatter(x="station_london", y="station_paris", alpha=0.5)
    plt.show()
    print([
    method_name
    for method_name in dir(air_quality.plot)
        if not method_name.startswith("_")])

    air_quality.plot.box()
    plt.show()

    air_quality.plot.hist()
    plt.show()

    air_quality["station_paris"].plot.hist()
    plt.show()

    axs = air_quality.plot.area(figsize=(12, 4), subplots=True)
    plt.show()

    fig, axs = plt.subplots(figsize=(12, 4))
    air_quality.plot.area(ax=axs)
    axs.set_ylabel("NO$_2$ concentration")
    fig.savefig("no2_concentrations.png")
    plt.show()

def pandatut05():
    air_quality = pd.read_csv("data/air_quality_no2.csv", index_col=0, parse_dates=True)
    print(air_quality.head())

    air_quality["london_mg_per_cubic"] = air_quality["station_london"] * 1.882
    print(air_quality.head())

    air_quality["ratio_paris_antwerp"] = (air_quality["station_paris"] / air_quality["station_antwerp"])
    print(air_quality.head())

    air_quality_renamed = air_quality.rename(
        columns={
            "station_antwerp": "BETR801",
            "station_paris": "FR04014",
            "station_london": "London Westminster",   }  )
    print(air_quality_renamed.head())
    air_quality_renamed = air_quality_renamed.rename(columns=str.lower)
    print(air_quality_renamed.head())