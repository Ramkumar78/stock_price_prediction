# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Ramkumar78/stock_price_prediction/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                    |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------- | -------: | -------: | ------: | --------: |
| app/api/main.py                         |      134 |       68 |     49% |79-81, 88, 106-108, 112-136, 152, 155, 165-228, 235-249 |
| app/core/download\_data.py              |       48 |       30 |     38% |30-67, 81-90, 93 |
| app/core/feature\_engineering.py        |      113 |       70 |     38% |120-136, 150-171, 190-195, 217-231, 236-332 |
| app/core/price\_based\_features.py      |      110 |       41 |     63% |53, 56, 59, 63, 187-203, 208-221, 226-241, 250-269, 274-310 |
| app/core/regime\_dependent\_features.py |       72 |       16 |     78% |169, 178, 185-186, 191-215 |
| app/core/regime\_features.py            |       53 |       11 |     79% |   157-180 |
| app/core/technical\_features.py         |       75 |       16 |     79% |166, 175, 182-183, 188-212 |
| app/core/volatility\_features.py        |       33 |       11 |     67% |     73-96 |
| app/core/volume\_features.py            |       30 |       11 |     63% |    97-120 |
| **TOTAL**                               |  **668** |  **274** | **59%** |           |


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