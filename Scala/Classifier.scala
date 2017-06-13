/**
  * Created by bhumi on 6/11/2017.
  */
import scala.io.Source
import scala.util.matching.Regex._
import scala.util.Random
import scala.collection.immutable.ListMap

class Classifier {
  def preprocess (): (List[String], List[Int]) =  {

    val data = scala.io.Source.fromFile("Spam", "UTF8")
      .getLines.map(_.stripLineEnd.split("\t", -1))
      .map(fields => fields(0) -> fields(1)).toList
    var x = data(1)
    var num_sms = data.size
    var clean_data = List[String]()
    var labels = List[Int]()
    var i = 0
    for (i <- 0 to (num_sms-1)) {
      if (data(i)._1 == "spam") {
        labels = labels :+ 1
      }
      else {
        labels = labels :+ 0
      }
      var cleaner = List(clean(data(i)._2))

      clean_data = List.concat(clean_data,cleaner)
    }

    return (clean_data, labels)
  }

  def clean(text: String): String={

    val regex = "[^a-zA-Z\\s]".r

    val not_so_special = regex.replaceAllIn(text, "")
    val lower_case = not_so_special.toLowerCase()
    //val words = lower_case.split(" ").toList
    return lower_case
  }

  def split_and_featurize(data: List[String], labels: List[Int], max_feature:Int): (List[Int],List[Int],List[List[Int]],List[List[Int]]) = {

    var split_ratio = 0.9

    var train_data = data.take((data.size * 0.9).toInt)
    var test_data = data.drop((data.size * 0.9).toInt)

    var final_features = count_vector(train_data, max_feature)

    var features_train = featurize(train_data, max_feature,  final_features)
    var features_test = featurize(test_data, max_feature, final_features)
    return(labels.take((labels.size * 0.9).toInt), labels.drop((labels.size * 0.9).toInt), features_train, features_test)
    //print(final_features)
  }

  def featurize(data: List[String], max_feature:Int , final_features:List[String]): List[List[Int]] = {

    var data_features = List[List[Int]]()
    for (i <- 1 to data.size-1) {
      var words = (data(i)).split(" ").toList
      //print(words)
      var individual_sent_freq = words.groupBy(identity).mapValues(_.size)
      var feature = List.fill(max_feature)(0)
      var temp= feature.toArray
      for (j <- 1 to max_feature - 1) {
        if(words.contains(final_features(j)) ) {
          temp(j) = individual_sent_freq(final_features(j))
        }
        else {
          temp(j) = 0
        }
      }
      feature = temp.toList
      data_features = data_features :+ feature
    }


    return(data_features)

  }

  def count_vector(data: List[String], max_feature:Int): List[String] = {

    var all_words = List[String]()
    for (i <- 1 to data.size - 1) {
      var words = (data(i)).split(" ").toList
      all_words = List.concat(all_words, words)
    }
    //print(all_words)
    var freq = all_words.groupBy(identity).mapValues(_.size)

    //var max_feature = 1000
    freq = ListMap(freq.toSeq.sortWith(_._2 > _._2): _*)
    var frequency = freq.take(max_feature)
    var final_features = frequency.keySet.toList

    //print(frequency)

    return(final_features)
  }


}

object Classifier {
  def main (args: Array[String]): Unit = {

    var nb = new Classifier()
    var newer:(List[String], List[Int])= nb.preprocess()

    //shuffle data
    var zippedData = newer._1.zip(newer._2)
    //zippedData = Random.shuffle(zippedData)
    newer = zippedData.unzip
    var data = newer._1
    var labels = newer._2

    var max_feature = 1000
    var split_data:(List[Int],List[Int],List[List[Int]],List[List[Int]])=nb.split_and_featurize(data,labels,max_feature)
    var train_features = split_data._3
    var test_features = split_data._4
    var train_labels = split_data._1
    var test_labels = split_data._2

    //call the Multinomial library
    var model = new MultinomialNB()
    var (featureProb, priorProb) = model.fit(train_features, train_labels)
    var pred = model.predict(featureProb, priorProb, test_features)

    //calculate accuracy
    var accurateLabelCount = 0
    for(i <- 1 to pred.size - 1){
      //println(pred(i)==test_labels(i))
      if (pred(i) == test_labels(i)){
        accurateLabelCount += 1
      }
      //print(accurateLabelCount, pred.size)
    }

    var accuracy = (float2Float(accurateLabelCount)/pred.size) * 100

    print( accuracy)

  }
}


