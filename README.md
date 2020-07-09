# LocationTagger

> This repository provides a Location Tagger, for identifying locations, using a BERT-CRF Tagger. It creates a Location chunk using IOB tags when it finds one or more location words.

> This code is based on the [BiLSTM-CCM tagger](https://github.com/codedecde/BiLSTM-CCM) repository.

---

## Requirements

-   Python 3.4+
-   Linux-based system
---

## Installation

### Clone

> Clone this repository to your local machine.
```bash
git clone "https://github.com/dair-iitd/LocationTagger.git"
```

### Environment Setup

Please follow the instructions at the following link to set up anaconda. [https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)
> Set up the conda environment
```bash
$ conda env create -f environment.yml
```

> Install the required python packages

```bash
$ conda activate location-tagger
$ pip install -r requirements.txt
```
---

## Set up

### Stanford Core-NLP-Server
Please install the stanford core-nlp server library using the following link:
http://nlp.stanford.edu/software/stanford-corenlp-latest.zip
The server needs to run (on port 9000) when generating the features or outputs.

Use the following command to run the server.

```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000
```
To shut down the server use the following command:
```bash
wget "localhost:9000/shutdown?key=`cat /tmp/corenlp.shutdown`" -O -
```

### BERT

Download [https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt)
 and save as "data/utils/bert/vocab.txt".

Download [https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz) and save as "data/utils/bert/bert-base-multilingual-cased.tar.gz".

## Description

The repository can be used to tag locations in a tourism forum post (example shown below). The repository provides scripts for training/testing the model, changing model configurations, evalaute precision/recall of tagged locations given the gold data, and processing/filtering generic tagged locations like states, countries, acronyms, etc.

A sample input structure is shown below:
```json
{
	"question": "We will spend a weekend in October in NY as part of a longer trip to the US. We have both been in NY before and have a few places we wanna visit and re-visit but we are still searching for a nice seafood place for dinner. We have had a look at Luke's Lobster and Pearl Oyster Bar to name a few. . . Any comments on these two? Other recommendations for great seafood?",
	"url": "https://www.tripadvisor.in/ShowTopic-g28953-i4-k4852356-Seafood_place_in_NY-New_York.html"
}
```

The output file is shown below:
```json
{
	"question": "We will spend a weekend in October in NY as part of a longer trip to the US. We have both been in NY before and have a few places we wanna visit and re-visit but we are still searching for a nice seafood place for dinner. We have had a look at Luke's Lobster and Pearl Oyster Bar to name a few. . . Any comments on these two? Other recommendations for great seafood?",
	"url": "https://www.tripadvisor.in/ShowTopic-g28953-i4-k4852356-Seafood_place_in_NY-New_York.html",
	"question_pos": "We_0 will_1 spend_2 a_3 weekend_4 in_5 October_6 in_7 NY_8 as_9 part_10 of_11 a_12 longer_13 trip_14 to_15 the_16 US_17 ._18 We_19 have_20 both_21 been_22 in_23 NY_24 before_25 and_26 have_27 a_28 few_29 places_30 we_31 wan_32 na_33 visit_34 and_35 re-visit_36 but_37 we_38 are_39 still_40 searching_41 for_42 a_43 nice_44 seafood_45 place_46 for_47 dinner_48 ._49 We_50 have_51 had_52 a_53 look_54 at_55 Luke_56 's_57 Lobster_58 and_59 Pearl_60 Oyster_61 Bar_62 to_63 name_64 a_65 few_66 ..._67 Any_68 comments_69 on_70 these_71 two_72 ?_73 Other_74 recommendations_75 for_76 great_77 seafood_78 ?_79",
	"tagged_location": [
	    "Luke_56 's_57 Lobster_58",
	    "Pearl_60 Oyster_61 Bar_62"
    ]
}
```

## Generating Features

The features generator has two possible options for different file formats.

For the annotation file format, use the following command,

```bash
python -m utils.generateFeatureFileFromAnnotations --input_file_path "data/inputs/$1" --features_file_path "data/features/$2$
```

For the typical file (as shown in description), use the following command,

```bash
python -m utils.generateFeatureFileForTourqueData --data_file_path "data/inputs/$1" --features_file_path "data/features/$2$
```

## Training

```bash
$ python -m src.main --train --num_epochs 5 --data_file_path "data/features/train_annotations.features.txt" --serialization_dir "data/models" --pretrained_model_path "data/models/best.weights" --config_file "data/configs/config.jsonnet" --devices 0
```

The config.jsonnet file can be changed as per the user requirements.
pretrained_model_path need not be specified.

## Testing

```bash
$ python -m src.main --test --data_file_path "data/features/validation_questions.features.txt" --predictions_file_path "data/predictions/validation_questions.predictions.txt" --"pretrained_model_path data/models/best.weights" --config_file "data/configs/config.jsonnet" --devices 0   
```

## Evaluate Predictions

All feature files contain labels. For annotation files, the labels can be "O", "B-GPE", "I-GPE". However, for other files, only "O" labels are present. Use the following command to find precision/recall of tagged locations.

```bash
python -m analysis.evaluate --prediction_file_path "data/predictions/test_annotations.predictions.txt" --gold_file_path "data/features/test_annotations.features.txt"
```

## Generate Outputs

The predictions file contains only the tags like "B-GPE", "I-GPE", "O". The following command can be used to parse the location tokens by generating a conll tree.

```bash
python -m utils.generateSanitizedPredictions --input_file_path "data/features/train_questions.json" --features_file_path "data/features/train_questions.features.txt" --predictions_file_path "data/features/train_questions.predictions.txt" --output_file_path "data/features/train_annotations.features.txt" --process
```

The process argument is optional. It used to filter out countries, states, acronyms, etc. Note that repetitions may occur in tagged locations (different token positions).

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)

- **[Apache 2.0 license](https://opensource.org/licenses/Apache-2.0)**
