# Children's Book Cover Analysis

This project aims to analyze the visual and textual features of children's book covers in order to understand how they contribute to the recommendation of books for different age groups. The project is divided into three parts: Visual Attributes, Object Detection, and Implied Story.

### Visual Attributes
The visual analysis of the book covers is done using the Python Imagine Library OpenCV. The brightness, colorfulness and entropy values of the covers are extracted and analyzed across different age groups. Additionally, the dominant colors present in the covers are also analyzed to understand the patterns of color usage.

### Object Detection
Zero-shot dection is used to extract the objects in the cover images. The detection is done using the Transformers library's. The individual objects are analyzed per age group to test for statistical significance across age groups.

### Implied Story
Using the BLIP Captioning Model, captions are generated for all voers. These captions are analyzed using the libraries spaCy for Named Entity Recognition (NER) and Gensim for Topic Modeling. The captions of the book covers are analyzed to understand the themes and topics present in the captions across different age groups and sources.

## Usage
Make sure to first install the requirements and the ```en_core_web_sm``` by running
```Python
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

There are also two ```nltk.download``` functions in ```main.py``` which may be commented out after the first run.

## Input
Supply the books the [book_data.json](/data/book_data.json) file, or create a new file (just change the ```input_path``` variable to the filepath).

Make sure the input data has the following format:
```json
[
    {
        "cover": "<URL>",
        "age": [
            1,
            2
        ]
    }
]
```
with the ```cover```field containing the URL to the cover image, and the ```age``` field containg a list of the age groups the book is appropriate for.
Note that the project only returns accurate results if the dataset has a significant amount of books.

Finally, run the code by navigating to the root of the project directory and running
```Python
python main.py
```

This will print the results in the command line, as well as save them to ```data/stat_result.txt``` (or the specified path in ```stat_res_path```).

## Output
The project consists of two separate outputs: a ```json``` file that contains the extracted results from all three aspects of book covers, and a ```txt``` file that contains the analytical results from the extracted data.
It is recommended the command line be used to view individual results. Note that if the data set is small, some statistical models might not work and will give warning in the command line.

## Conclusion
The project provides insights into the visual features of children's book covers and how they contribute to the recommendation of books for different age groups. Preliminary analysis shows statistical significance across different age groups in all three aspects.
