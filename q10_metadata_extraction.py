# Q10: Metadata Extraction using PySpark

from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
from pyspark.sql import functions as F
from pyspark.sql.functions import regexp_extract, desc, length, avg

# Create Spark Session 

spark = SparkSession.builder \
    .appName("Q10_Metadata_Extraction") \
    .master("local[*]") \
    .getOrCreate()



# Read all text files from local directory
books_df = spark.read.text("D184MB/*.txt") \
    .withColumn("file_path", input_file_name())

# Extract file name from path
books_df = books_df.withColumn(
    "file_name",
    F.regexp_extract("file_path", r"([^/]+$)", 1)
)

# Combine all lines of each book into one document
books_df = books_df.groupBy("file_name") \
    .agg(F.concat_ws("\n", F.collect_list("value")).alias("text"))

# Extract Metadata using Regular Expressions
title_regex = r"Title:\s*(.*)"
date_regex = r"Release Date:\s*(.*)"
language_regex = r"Language:\s*(.*)"
encoding_regex = r"Character set encoding:\s*(.*)"

books_df = books_df \
    .withColumn("title", regexp_extract("text", title_regex, 1)) \
    .withColumn("release_date", regexp_extract("text", date_regex, 1)) \
    .withColumn("language", regexp_extract("text", language_regex, 1)) \
    .withColumn("encoding", regexp_extract("text", encoding_regex, 1))

# Show extracted metadata
print("\nExtracted Metadata:")
books_df.select("file_name", "title", "release_date", "language", "encoding") \
    .show(10, truncate=False)


# Extract Year from Release Date
books_df = books_df.withColumn(
    "year",
    regexp_extract("release_date", r"(\d{4})", 1)
)

print("\nBooks with Extracted Year:")
books_df.select("file_name", "release_date", "year") \
    .show(10, truncate=False)

# Number of Books per Year
books_per_year = books_df.groupBy("year") \
    .count() \
    .orderBy("year")

print("\nNumber of Books per Year:")
books_per_year.show()


# Most Common Language

language_stats = books_df.groupBy("language") \
    .count() \
    .orderBy(desc("count"))

print("\nMost Common Languages:")
language_stats.show()


# Average Title Length
print("\nAverage Title Length:")
books_df.select(
    avg(length("title")).alias("average_title_length")
).show()


spark.stop()
