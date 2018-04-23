# text-mining
Unstructured Data Analysis (Graduate) @Korea University

## Notice
* Term Project Group
  * 1조: 손규빈	박경찬	최희정
  * 2조: 이재융	김혜민	이수연
  * 3조:	이주현	권원진	정하은 최현석		
  * 4조:	음수민	유지원	김웅	전은석	권구포
  * 5조:	이도명	김강민	이중호	김다애	
  * 6조:	채선율	성유연	이창현		
  * 7조:	안지영	송재승	김문수	최지은	
  * 8조:	최현율	황정임	조억 김명소	
  * 9조:	이민형	이선화	손주희	김준호	
  * 10조: 김다연	윤석채	우현희	안건이	이지윤
  * 11조: 송서하	양우식	정민성					
  * 12조:	김동원 박재용 정연재
* Term project proposal evaluation ([link](https://goo.gl/forms/iDDZoKTPCxiOZItb2))
* Term project proposal feedback ([download](https://www.dropbox.com/s/lbi9vp824nxel2i/2018_Term%20Project_%EC%A0%9C%EC%95%88%EB%B0%9C%ED%91%9C%20%ED%94%BC%EB%93%9C%EB%B0%B1.xlsx?dl=0))

* Recommended courses
  * CS224d @Stanford: Deep Learning for Natural Language Processing
    * Course Homepage: http://cs224d.stanford.edu/
    * YouTube Video: https://www.youtube.com/playlist?list=PLlJy-eBtNFt4CSVWYqscHDdP58M3zFHIG
  * CS224n @Stanford: Natural Language Processing Deep Learning
    * Course Homepage: http://web.stanford.edu/class/cs224n/syllabus.html
    * Youtube Video: https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6
  * Deep Natural Lanugage Processing @Oxford
    * Course Homepage: https://github.com/oxford-cs-deepnlp-2017/lectures

## Schedule
## Topic 1: Introduction to Text Mining
* The usefullness of large amount of text data and the challenges
* Overview of text mining methods

## Topic 2: From Texts to Data
* Obtain texts to analyze
* Text data collection through APIs and web scraping

## Topic 3: Natural Language Processing
* Introduction to NLP
* Lexical analysis
* Syntax analysis
* Other topics in NLP
* Reading materials
  * Cambria, E., & White, B. (2014). Jumping NLP curves: A review of natural language processing research. IEEE Computational intelligence magazine, 9(2), 48-57. ([PDF](http://ieeexplore.ieee.org/abstract/document/6786458/))
  * Collobert, R., Weston, J., Bottou, L., Karlen, M., Kavukcuoglu, K., & Kuksa, P. (2011). Natural language processing (almost) from scratch. Journal of Machine Learning Research, 12(Aug), 2493-2537. ([PDF](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf))
  * Young, T., Hazarika, D., Poria, S., & Cambria, E. (2017). Recent trends in deep learning based natural language processing. arXiv preprint arXiv:1708.02709. ([PDF](https://arxiv.org/pdf/1708.02709.pdf))

## Topic 4-1: Document Representation I: Classic Methods
* Bag of words
* Word weighting
* N-grams

## Topic 4-2: Document Representation II: Distributed Representation
* Word2Vec
* GloVe
* FastText
* Doc2Vec
* Reading materials
  * Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. Journal of machine learning research, 3(Feb), 1137-1155. ([PDF](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf))
  * Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781. ([PDF](https://arxiv.org/pdf/1301.3781.pdf))
  * Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems (pp. 3111-3119). ([PDF](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf))
  * Pennington, J., Socher, R., & Manning, C. (2014). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543). ([PDF](http://www.aclweb.org/anthology/D14-1162))
  * Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2016). Enriching word vectors with subword information. arXiv preprint arXiv:1607.04606. ([PDF](https://arxiv.org/pdf/1607.04606.pdf))

## Topic 5: Dimensionality Reduction
* Dimensionality Reduction
* Supervised Feature Selection
* Unsupervised Feature Extraction: Latent Semantic Analysis (LSA) and t-SNE
* R Example
* Reading materials
  * Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by latent semantic analysis. Journal of the American society for information science, 41(6), 391. ([PDF](http://lsa.colorado.edu/papers/JASIS.lsi.90.pdf))
  * Dumais, S. T. (2004). Latent semantic analysis. Annual review of information science and technology, 38(1), 188-230.
  * Maaten, L. V. D., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(Nov), 2579-2605. ([PDF](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)) ([Homepage](https://lvdmaaten.github.io/tsne/))

## Topic 6: Document Similarity & Clustering
* Document similarity metrics
* Clustering overview
* K-Means clustering
* Hierarchical clustering
* Density-based clustering
* Reading materials
  * Jain, A. K., Murty, M. N., & Flynn, P. J. (1999). Data clustering: a review. ACM computing surveys (CSUR), 31(3), 264-323. ([PDF](http://dataclustering.cse.msu.edu/papers/MSU-CSE-00-16.pdf))

## Topic 7-1: Topic Modeling I
* Topic modeling overview
* Probabilistic Latent Semantic Analysis: pLSA
* LDA: Document Generation Process
* Reading materials
  * Hofmann, T. (1999, July). Probabilistic latent semantic analysis. In Proceedings of the Fifteenth conference on Uncertainty in artificial intelligence (pp. 289-296). Morgan Kaufmann Publishers Inc. ([PDF](http://www.iro.umontreal.ca/~nie/IFT6255/Hofmann-UAI99.pdf))
  * Hofmann, T. (2017, August). Probabilistic latent semantic indexing. In ACM SIGIR Forum (Vol. 51, No. 2, pp. 211-218). ACM.
  * Blei, D. M. (2012). Probabilistic topic models. Communications of the ACM, 55(4), 77-84. ([PDF](http://delivery.acm.org/10.1145/2140000/2133826/p77-blei.pdf?ip=175.114.11.68&id=2133826&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1524148444_1a7687d674528eeabc9a97afa2db5a29))
  * Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022. ([PDF](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf))

## Topic 7-2: Topic Modeling II
* LDA Inference: Gibbs Sampling
* LDA Evaluation
* Recommended video lectures
  * LDA by D. Blei ([Lecture Video](http://videolectures.net/mlss09uk_blei_tm/))
  * Variational Inference for LDA by D. Blei ([Lecture Video](https://www.youtube.com/watch?v=Dv86zdWjJKQ&t=113s))

## Topic 8: Document Classification I
* Document classification overview
* Naive Bayesian classifier
* k-Nearest Neighbor classifier
* Classification tree
* Support Vector Machine (SVM)

## Topic 9: Document Classification II
* Introduction to Neural Network
* Recurrent neural network-based document classification
* Convolutional neural network-based document classification

