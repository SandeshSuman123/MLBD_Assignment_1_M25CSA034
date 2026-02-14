# Q12 Author Influence Network using Local Spark Setup

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col
import pyspark.sql.functions as F
import os
from pyspark.sql import Row


spark = SparkSession.builder \
    .appName("Q12_Author_Influence_Network") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

print("Spark Version:", spark.version)

# load only first 30 books
files = sorted(os.listdir("D184MB"))[:30]

books = []

for file in files:
    if file.endswith(".txt"):
        with open(f"D184MB/{file}", "r", encoding="utf-8", errors="ignore") as f:
            books.append(Row(file_name=file, text=f.read()))

books_df = spark.createDataFrame(books)

print("Total Books Loaded:", books_df.count())
books_df.show(5)

# extract author
books_df = books_df.withColumn(
    "author",
    regexp_extract("text", r"Author:\s*(.*)", 1)
)

# extract release date 
books_df = books_df.withColumn(
    "release_date",
    regexp_extract("text", r"Release Date:\s*(.*)", 1)
)

# extract year 
books_df = books_df.withColumn(
    "year",
    regexp_extract("release_date", r"(\d{4})", 1).cast("int")
)

books_df.select("file_name", "author", "year").show(10, truncate=False)

# define time window
X = 5

# create pairwise author comparison
df1 = books_df.select("author", "year").alias("df1")
df2 = books_df.select("author", "year").alias("df2")

edges = df1.crossJoin(df2) \
    .where(
        (F.col("df1.author") != F.col("df2.author")) &
        (F.col("df1.year").isNotNull()) &
        (F.col("df2.year").isNotNull()) &
        (F.col("df2.year") - F.col("df1.year") > 0) &
        (F.col("df2.year") - F.col("df1.year") <= X)
    ) \
    .select(
        F.col("df1.author").alias("author1"),
        F.col("df2.author").alias("author2")
    )

edges.show(10, truncate=False)

# compute out-degree 
out_degree = edges.groupBy("author1") \
    .count() \
    .withColumnRenamed("count", "out_degree")

out_degree.orderBy(F.desc("out_degree")).show(5)

# compute in-degree 
in_degree = edges.groupBy("author2") \
    .count() \
    .withColumnRenamed("count", "in_degree")

in_degree.orderBy(F.desc("in_degree")).show(5)


spark.stop()
