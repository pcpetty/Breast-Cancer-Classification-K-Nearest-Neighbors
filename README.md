# Breast Cancer Classification using K-Nearest Neighbors (KNN)

## Project Overview

This project leverages the K-Nearest Neighbors (KNN) algorithm to classify breast cancer tumors as malignant or benign based on clinical data. The objective is to develop a robust and interpretable model that can support medical professionals in making accurate diagnoses. By exploring this dataset, we aim to uncover insights into the key factors that contribute to the classification of cancerous tumors, providing both statistical rigor and practical implications for healthcare applications.


## Objectives

1. **Data Exploration**: Analyze the dataset to understand the distribution of features and target classes.
2. **Data Preprocessing**: Scale and prepare the dataset for model training.
3. **Model Training and Tuning**: Implement the KNN classifier and tune hyperparameters to optimize performance.
4. **Model Evaluation**: Evaluate the model using accuracy, precision, recall, and F1-score.
5. **Visualizations and Interpretations**: Create visualizations to interpret the results and model behavior.
6. **Conclusions**: Summarize the findings and suggest potential improvements for future work.

## Dataset

The dataset used for this project is the Breast Cancer Wisconsin dataset, which is available in the `sklearn.datasets` module. It contains 30 numerical features representing characteristics of the cell nuclei.

## Dependencies

- Python 3.7+
- Jupyter Notebook
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

Install the necessary libraries using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to Run

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Open the Jupyter Notebook file `Enhanced_Cancer_Classifier_Project.ipynb`.
4. Run the cells sequentially to execute the entire workflow.

## Results

The final model achieves an accuracy of `92%` on the test set, with a balanced performance across other evaluation metrics such as precision, recall, and F1-score.

## Visualizations

Visualizations included in the notebook help to understand:
- Feature importance and distribution.
- Model performance and accuracy as a function of the number of neighbors (k).
- Confusion matrix to visualize classification results.


## Conclusion

The KNN classifier demonstrates a solid performance in classifying breast cancer tumors, with high accuracy and balanced metrics across the board. However, as with any model, there is room for improvement. Future work could explore advanced algorithms or integrate domain-specific knowledge to enhance predictive accuracy. The results of this project underline the potential of machine learning in augmenting clinical decision-making processes, highlighting the importance of data-driven approaches in modern healthcare.


## Acknowledgments

This project is inspired by real-world applications of machine learning in healthcare. Special thanks to the contributors of the original dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
