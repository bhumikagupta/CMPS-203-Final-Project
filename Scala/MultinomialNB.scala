/**
  * Created by aishnaagrawal on 6/10/17.
  */



class MultinomialNB {

  private var _alpha = 1

  def alpha = _alpha

  def fit(X: List[List[Int]], y: List[Int]): (List[List[Double]], List[Double]) = {
    var classes = y.toSet
    var zippedData = X zip y
    var separated = List[List[List[Int]]]()
    val count_sample = double2Double(X.size)

    for (cat <- classes) {
      separated = separated :+ seperateByCategory(zippedData, cat)
    }

    //Find prior log probabilities
    var priorLogProb = List[Double]()
    for (i <- separated) {
      priorLogProb = priorLogProb :+ math.log(i.size / count_sample)
    }

    //Count each word for each class and add alpha as smoothing
    var countList = List[List[Int]]()

    for (i <- separated) {
      var categoryCount = List.fill(i(0).size)(alpha)
      for (j <- i) {
        categoryCount = categoryCount.zip(j).map(x => x._1 + x._2)
      }
      countList = countList :+ categoryCount

    }

    //calculate log probability of each word
    var featureLogProb = List[List[Double]]()
    for (i <- countList) {
      var categoryFeatureLogProb = List[Double]()
      for (j <- i) {
        categoryFeatureLogProb = categoryFeatureLogProb :+ math.log(j / double2Double(i.sum))
      }
      featureLogProb = featureLogProb :+ categoryFeatureLogProb
    }
    return (featureLogProb, priorLogProb)
  }

  def seperateByCategory(X: List[(List[Int], Int)], cat: Int): List[List[Int]] = {
    var subset = List[List[Int]]()
    for (feature <- X) {
      if (feature._2 == cat) {
        subset = subset :+ feature._1
      }
    }
    return subset
  }


  def predict(featureProb: List[List[Double]], priorProb: List[Double], testData: List[List[Int]]): List[Int] = {
    var prediction = List[Int]()
    var postProb = List[List[Double]]()

    for(i <- testData){
      var rowPostProb = List[Double]()
      for (j <- featureProb){
        rowPostProb = rowPostProb :+ i.zip(j).map(x => x._1 * x._2).sum
      }
      rowPostProb = rowPostProb.zip(priorProb).map(y => y._1 + y._2)
      rowPostProb = rowPostProb.reverse
      postProb = postProb :+ rowPostProb
    }

    var finalPred = List[Int]()
    for(i<-postProb){
      finalPred = finalPred :+ i.zipWithIndex.maxBy(_._1)._2
    }

    return finalPred
  }

}

//
//object MultinomialNB{
//  def main(args: Array[String]): Unit = {
//    var model = new MultinomialNB()
//    var x = List(List(1,0,0,0,1,1), List(2,1,0,0,0,0),List(2,0,1,0,0,0),List(1,0,0,1,0,0))
//    var y = List(1, 0, 0, 0)
//    var x_test = List(List(3,0,0,0,1,1),List(0,1,1,0,1,1))
//    var (featureProb, priorProb) = model.fit(x, y)
//    var pred = model.predict(featureProb, priorProb, x_test)
//    print(pred)
//  }
//}
