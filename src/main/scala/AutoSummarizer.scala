import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{Tokenizer, Word2Vec}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.functions._
import org.apache.spark.{SparkConf, SparkContext}

object Main extends App {
  new AutoSummarizer("/home/whisper/Desktop/doc")
    .summarize(inOrder = true).foreach(println)
}

trait SparkConfig {
  protected val conf = new SparkConf()
    .setAppName("AutoSummarizer")
    .setMaster("local[*]")
  protected val sc = new SparkContext(conf)
  sc.setLogLevel("ERROR")

  protected val sqlContext = new org.apache.spark.sql.SQLContext(sc)

}

trait Summarizer {
  def summarize(count: Int, inOrder: Boolean): List[String]
}

class AutoSummarizer(path: String) extends SparkConfig with Summarizer {

  override def summarize(count: Int = 5, inOrder: Boolean = false): List[String] = {
    import sqlContext.implicits._
    val pattern = "([A-Z][^\\.!?]*[\\.!?])".r
    val sentences = sc.textFile(path).collect().mkString
    val sentencesRDD = sc.parallelize(
      pattern
        .findAllIn(sentences)
        .toArray
        .map(_.toString)
    ).zipWithIndex()
    val sentencesDF = sentencesRDD.toDF("sentences", "index").cache()

    println(s"the document has ${sentencesDF.count()} sentences\n")

    val tokenizer = new Tokenizer()
      .setInputCol("sentences")
      .setOutputCol("words")

    val word2Vec = new Word2Vec()
      .setInputCol("words")
      .setOutputCol("vectors")
      .setVectorSize(1000)
      .setMinCount(0)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, word2Vec))

    val model = pipeline.fit(sentencesDF)
    val result = model.transform(sentencesDF)
    val vec = result.select("vectors").map(_ (0).asInstanceOf[Vector])
    val documentVector = Statistics.colStats(vec).mean

    val cos_sim = { (a: Vector) =>
      def dotProduct(a: Vector, b: Vector) = {
        a.toArray zip b.toArray map { case (a, b) => a * b } sum
      }
      dotProduct(a, documentVector) / (math.sqrt(dotProduct(a, a)) * math.sqrt(dotProduct(documentVector, documentVector)))
    }
    val sqlFunc = udf(cos_sim)
    result.withColumn("sim", sqlFunc(col("vectors")))
      .orderBy(desc("sim"))
      .select("sentences", "index")
      .take(5)
      .map(row => (row(0) toString, row(1).asInstanceOf[Long]))
      .sortBy(_._2)
      .map(_._1).toList
  }
}
