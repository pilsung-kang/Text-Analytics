# text-mining
Unstructured Data Analysis (Graduate) @Korea University

## Notice
* **Syllabus** ([download](https://github.com/pilsung-kang/text-mining/blob/master/2019_Spring_Unstructured%20Data%20Analysis.pdf))
* **Term project groups**
  * 1조: 박성훈, 이수빈(2018021120), 이준걸, 박혜준
  * 2조: 이정호, 천우진, 유초롱, 조규원
  * 3조: 백승호, 목충협, 변준형, 이영재
  * 4조: 박건빈, 이수빈(2018020530), 변윤선, 권순찬
  * 5조: 최종현, 이정훈, 박중민, 노영빈
  * 6조: 백인성, 김은비, 신욱수, 강현규
  * 7조: 전성찬, 박현지, 문관영
  * 8조: 조용원, 정승섭, 민다빈, 최민서
  * 9조: 박명현, 장은아, 유건령 
* **Term project proposal**
  * [Evaluation](https://forms.gle/YhiziBFyDZX68uzr7) (due: 4/4 Thu. 23:59)
  * [Comments by the lecturer](https://www.dropbox.com/s/tn6a17yp5bz0yq3/Term%20Project%20Proposal%20%ED%8F%89%EA%B0%80%ED%91%9C_%EB%B0%B0%ED%8F%AC%EC%9A%A9.pdf?dl=0)
  * [Comments by students](https://www.dropbox.com/s/n8sarx07n4xs54s/2019%20Term%20Project%20Proposal%20Evaluation%20%28Responses%29_upload.xlsx?dl=0)
* **Term project inteim presentation**
  * [Presentation Slides](https://www.dropbox.com/s/x34lwhqxej70e07/%EC%A4%91%EA%B0%84%EB%B0%9C%ED%91%9C.zip?dl=0)
  * [Evaluation](https://forms.gle/cKtSeCVtNR7yFQGn8) (due: 5/21 Tue. 23:59)
  * [Comments by the lecturer](https://www.dropbox.com/s/hkp9rxpdr67dfu7/Term%20Project%20%EC%A4%91%EA%B0%84%EB%B0%9C%ED%91%9C%20%ED%8F%89%EA%B0%80%ED%91%9C_%EB%B0%B0%ED%8F%AC%EC%9A%A9.pdf?dl=0)
  * [Comments by students](https://www.dropbox.com/s/vknfuooy7gx0afn/2019%20Term%20Project%20Interim%20Presentation%20Evaluation%20%28Responses%29_upload.xlsx?dl=0)
* **Final Exam**
  * Date and Place: 2019-06-20 15:30~17:30, New Engineering Hall 218/224 ([download](https://www.dropbox.com/s/keg4tjtw4uiu8u8/2019_%EB%B9%84%EC%A0%95%ED%98%95%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D_%EC%84%B1%EC%A0%81.xlsx?dl=0))
  * A non-programmable calculator is allowed, smart phones must be turned off
  * A hand-written cheating paper (A4 size, 3 pages, back and forth) is allowed


## Recommended courses
  * CS224d @Stanford: Deep Learning for Natural Language Processing
    * Course Homepage: http://cs224d.stanford.edu/
    * YouTube Video: https://www.youtube.com/playlist?list=PLlJy-eBtNFt4CSVWYqscHDdP58M3zFHIG
  * CS224n @Stanford: Natural Language Processing Deep Learning
    * Course Homepage: http://web.stanford.edu/class/cs224n/syllabus.html
    * Youtube Video: https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6
  * Deep Natural Lanugage Processing @Oxford
    * Course Homepage: https://github.com/oxford-cs-deepnlp-2017/lectures

## Schedule
## Topic 1: Introduction to Text Analytics
* The usefullness of large amount of text data and the challenges
* Overview of text analytics methods

## Topic 2: From Texts to Data
* Text data collection: Web scraping

## Topic 3: Text Preprocessing
* Introduction to Natural Language Processing (NLP)
* Lexical analysis
* Syntax analysis
* Other topics in NLP
* Reading materials
  * Cambria, E., & White, B. (2014). Jumping NLP curves: A review of natural language processing research. IEEE Computational intelligence magazine, 9(2), 48-57. ([PDF](http://ieeexplore.ieee.org/abstract/document/6786458/))
  * Collobert, R., Weston, J., Bottou, L., Karlen, M., Kavukcuoglu, K., & Kuksa, P. (2011). Natural language processing (almost) from scratch. Journal of Machine Learning Research, 12(Aug), 2493-2537. ([PDF](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf))
  * Young, T., Hazarika, D., Poria, S., & Cambria, E. (2017). Recent trends in deep learning based natural language processing. arXiv preprint arXiv:1708.02709. ([PDF](https://arxiv.org/pdf/1708.02709.pdf))

## Topic 4: Neural Networks Basics
* Perception, Multi-layered Perceptron
* Convolutional Neural Networks (CNN)
* Recurrent Neural Networks (RNN)
* Practical Techniques

## Topic 5-1: Document Representation I: Classic Methods
* Bag of words
* Word weighting
* N-grams

## Topic 5-2: Document Representation II: Distributed Representation
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

## Topic 6: Dimensionality Reduction
* Dimensionality Reduction
* Supervised Feature Selection
* Unsupervised Feature Extraction: Latent Semantic Analysis (LSA) and t-SNE
* R Example
* Reading materials
  * Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by latent semantic analysis. Journal of the American society for information science, 41(6), 391. ([PDF](http://lsa.colorado.edu/papers/JASIS.lsi.90.pdf))
  * Dumais, S. T. (2004). Latent semantic analysis. Annual review of information science and technology, 38(1), 188-230.
  * Maaten, L. V. D., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(Nov), 2579-2605. ([PDF](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)) ([Homepage](https://lvdmaaten.github.io/tsne/))

## Topic 7: Document Similarity & Clustering
* Document similarity metrics
* Clustering overview
* K-Means clustering
* Hierarchical clustering
* Density-based clustering
* Reading materials
  * Jain, A. K., Murty, M. N., & Flynn, P. J. (1999). Data clustering: a review. ACM computing surveys (CSUR), 31(3), 264-323. ([PDF](http://dataclustering.cse.msu.edu/papers/MSU-CSE-00-16.pdf))

## Topic 8-1: Topic Modeling I
* Topic modeling overview
* Probabilistic Latent Semantic Analysis: pLSA
* LDA: Document Generation Process
* Reading materials
  * Hofmann, T. (1999, July). Probabilistic latent semantic analysis. In Proceedings of the Fifteenth conference on Uncertainty in artificial intelligence (pp. 289-296). Morgan Kaufmann Publishers Inc. ([PDF](http://www.iro.umontreal.ca/~nie/IFT6255/Hofmann-UAI99.pdf))
  * Hofmann, T. (2017, August). Probabilistic latent semantic indexing. In ACM SIGIR Forum (Vol. 51, No. 2, pp. 211-218). ACM.
  * Blei, D. M. (2012). Probabilistic topic models. Communications of the ACM, 55(4), 77-84. ([PDF](http://delivery.acm.org/10.1145/2140000/2133826/p77-blei.pdf?ip=175.114.11.68&id=2133826&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1524148444_1a7687d674528eeabc9a97afa2db5a29))
  * Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022. ([PDF](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf))

## Topic 8-2: Topic Modeling II
* LDA Inference: Gibbs Sampling
* LDA Evaluation
* Recommended video lectures
  * LDA by D. Blei ([Lecture Video](http://videolectures.net/mlss09uk_blei_tm/))
  * Variational Inference for LDA by D. Blei ([Lecture Video](https://www.youtube.com/watch?v=Dv86zdWjJKQ&t=113s))

## Topic 9: Document Classification
* Document classification overview
* Naive Bayesian classifier
* RNN-based document classification
* CNN-based document classification
* Reading materials
  * Kim, Y. (2014). Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882. ([PDF](http://www.aclweb.org/anthology/D14-1181))
  * Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. In Advances in neural information processing systems (pp. 649-657) ([PDF](https://arxiv.org/pdf/1509.01626.pdf))
  * Lee, G., Jeong, J., Seo, S., Kim, C, & Kang, P. (2018). Sentiment classification with word localization based on weakly supervised learning with a convolutional neural network. Knowledge-Based Systems, 152, 70-82. ([PDF](https://www.sciencedirect.com/science/article/pii/S0950705118301710))
  * Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 1480-1489). ([PDF](http://www.aclweb.org/anthology/N16-1174))
  * Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473. ([PDF](https://arxiv.org/pdf/1409.0473.pdf))
  * Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025. ([PDF](https://arxiv.org/pdf/1508.04025.pdf))

## Topic 10: Sentiment Analysis
* Architecture of sentiment analysis
* Lexicon-based approach
* Machine learning-based approach
* Reading materials
  * Hamilton, W. L., Clark, K., Leskovec, J., & Jurafsky, D. (2016, November). Inducing domain-specific sentiment lexicons from unlabeled corpora. In Proceedings of the Conference on Empirical Methods in Natural Language Processing. Conference on Empirical Methods in Natural Language Processing (Vol. 2016, p. 595). NIH Public Access. ([PDF](https://nlp.stanford.edu/pubs/hamilton2016inducing.pdf))
  * Zhang, L., Wang, S., & Liu, B. (2018). Deep learning for sentiment analysis: A survey. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 8(4), e1253. ([PDF](https://arxiv.org/ftp/arxiv/papers/1801/1801.07883.pdf))



