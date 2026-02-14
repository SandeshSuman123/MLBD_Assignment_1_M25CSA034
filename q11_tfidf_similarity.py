# Q11 TF-IDF and Book Similarity using Local Spark Setup

from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, regexp_replace, col
from pyspark.sql import functions as F
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Normalizer
from pyspark.sql.types import DoubleType
import os
from pyspark.sql import Row

# create spark session in local mode only
spark = SparkSession.builder \
    .appName("Q11_TFIDF_Similarity") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()


# load only first 20 books from  D184MB 
files = sorted(os.listdir("D184MB"))[:20]

books = []

for file in files:
    if file.endswith(".txt"):
        with open(f"D184MB/{file}", "r", encoding="utf-8", errors="ignore") as f:
            books.append(Row(file_name=file, text=f.read()))

books_df = spark.createDataFrame(books)

print("Total Books Loaded:", books_df.count())
books_df.show(5)

# clean text by removing punctuation and converting to lowercase 
books_df = books_df.withColumn(
    "clean_text",
    lower(regexp_replace("text", "[^a-zA-Z\\s]", " "))
)

# tokenize text into words 
tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
books_df = tokenizer.transform(books_df)

# remove stopwords 
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
books_df = remover.transform(books_df)

books_df.select("file_name", "filtered_words").show(2, truncate=False)

# compute term frequency using hashing tf 
hashingTF = HashingTF(
    inputCol="filtered_words",
    outputCol="rawFeatures",
    numFeatures=500
)

featurizedData = hashingTF.transform(books_df)

# compute idf
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
tfidfData = idfModel.transform(featurizedData)

tfidfData.select("file_name", "features").show(5, truncate=False)

# normalize tfidf vectors
normalizer = Normalizer(inputCol="features", outputCol="norm_features", p=2)
normalizedData = normalizer.transform(tfidfData)

# define cosine similarity udf 
def cosine_sim(v1, v2):
    return float(v1.dot(v2))

cosine_udf = F.udf(cosine_sim, DoubleType())

# compute pairwise similarity 
df1 = normalizedData.select("file_name", "norm_features").alias("df1")
df2 = normalizedData.select("file_name", "norm_features").alias("df2")

similarity_df = df1.crossJoin(df2) \
    .where(col("df1.file_name") != col("df2.file_name")) \
    .withColumn(
        "cosine_similarity",
        cosine_udf(col("df1.norm_features"), col("df2.norm_features"))
    ) \
    .select(
        col("df1.file_name").alias("book1"),
        col("df2.file_name").alias("book2"),
        "cosine_similarity"
    )

# show top 5 similar books
similarity_df.orderBy(col("cosine_similarity").desc()).show(5)
spark.stop()
