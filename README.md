# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Ramkumar78/stock_price_prediction/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                    |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------- | -------: | -------: | ------: | --------: |
| app/api/main.py                         |      149 |       23 |     85% |108-110, 117, 135-137, 181, 184, 208-209, 229-230, 237-246, 277-278 |
| app/core/custom\_pipeline.py            |       73 |       17 |     77% |37, 94, 116-128, 147-149, 155-156, 171-175 |
| app/core/download\_data.py              |       48 |       17 |     65% |41-42, 61-67, 81-90, 93 |
| app/core/feature\_engineering.py        |      113 |       60 |     47% |138, 150-171, 217-231, 236-332 |
| app/core/price\_based\_features.py      |      110 |       99 |     10% |43-65, 70-129, 134-182, 187-203, 208-221, 226-241, 250-269, 274-310 |
| app/core/regime\_dependent\_features.py |       72 |       16 |     78% |169, 178, 185-186, 191-215 |
| app/core/regime\_features.py            |       53 |       48 |      9% |32-128, 147-152, 157-180 |
| app/core/technical\_features.py         |       75 |       16 |     79% |166, 175, 182-183, 188-212 |
| app/core/volatility\_features.py        |       33 |       11 |     67% |     73-96 |
| app/core/volume\_features.py            |       30 |       11 |     63% |    97-120 |
| app/models/modelling\_catboost.py       |      275 |      275 |      0% |    63-676 |
| app/models/modelling\_ensemble.py       |      227 |      227 |      0% |    38-490 |
| app/models/modelling\_lightgbm.py       |      269 |      242 |     10% |114-134, 145-162, 181-223, 230-260, 267-316, 323-355, 388-464, 471-489, 496-510, 524-658, 662 |
| app/models/modelling\_xgboost.py        |      280 |      252 |     10% |139-159, 170-187, 206-248, 256-287, 294-346, 353-385, 418-496, 504-522, 529-544, 558-693, 698 |
| **TOTAL**                               | **1807** | **1314** | **27%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/Ramkumar78/stock_price_prediction/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/Ramkumar78/stock_price_prediction/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Ramkumar78/stock_price_prediction/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/Ramkumar78/stock_price_prediction/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FRamkumar78%2Fstock_price_prediction%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/Ramkumar78/stock_price_prediction/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.