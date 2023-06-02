# Dataset

- **Multi-GeNews**: Due to the lack of German multi-document summarization data, 
we have created a new mds dataset called 'Multi-GeNews'. This dataset contains news articles 
sourced from the news portal of SRF, a Swiss media company. The included articles span from January to March 2020.\
We're currently in the process of acquiring permission from SRF to share the full dataset publicly. In the meantime, we're able to share only the links to the original source articles. 
  - The dataset can be found under:```dataset/Multi-GeNews.jsonl```.
  - Each line of the jsonl file corresponds to a single cluster with the following json format:
    ```json
    {
      "articles": [
        {
          "title": "Title of Article 1",
          "text": "Text of Article 1",
          "article_link": "Link to Article 1"
        },
        {
          "title": "Title of Article 2",
          "text": "Text of Article 2",
          "article_link": "Link to Article 2"
        },
        ...
      ],
      "summary": "Summary of the cluster"
    }
    ```

  
