<!-- omit from toc -->
# Oasis Project

This project was developed for the "Alta Scuola Politecnica", a multidisciplinary honour program where students are initiated to a path of advanced interdisciplinary training to understand the complex rela‑
tionships between science, innovation, technology and socio‑economic systems, held between Politecnico di Milano and Politecnico di Torino in the academic years 2021-2022 and 2022-2023.

# Abstract

This report describes the development of the OASIS project, conducted by
students of the Alta Scuola Politecnica in collaboration with Atlante. The
project aims to provide a systematic and efficient approach to the deployment
of charging stations, thereby facilitating the transition towards a sustainable
transportation system that relies on EVs. By efficiently allocating charging
stations, this project seeks to enhance the accessibility and reliability of EV
charging infrastructure, making it convenient and easy for EV users to charge
their vehicles while also minimizing the impact on the environment. This
project is a research initiative whose goal is to implement a decisional algorithm
able to optimally locate and size electric vehicle charging stations. The work
is divided into two phases; the first phase involves conducting a comprehensive
literature review to explore the existing research on similar projects and to
identify any gaps in the current knowledge; the second phase consists of
implementing an algorithm to optimize the allocation of EV charging stations
based on various factors such as location, usage patterns, demand, and supply,
and developing an economic model to assess the profitability of the proposed
solution. The algorithm has been tested on different cities, on which it achieves
an optimal performance both for the sizing and allocation tasks. The results
show that the model identifies locations with a correct balance between the
different parameters and it attain high levels of station saturation and Return
on Investment, proving the financial robustness of the project. Overall, the
OASIS Project has the potential to transform the EV charging industry,
making it more sustainable, efficient, and accessible for everyone.

<!-- omit from toc -->
# Table of contents

- [Installation](#installation)
  - [How to clone the repository](#how-to-clone-the-repository)
  - [How to install the libraries](#how-to-install-the-libraries)
- [Analysis and results](#analysis-and-results)
- [Authors](#authors)

# Installation

## How to clone the repository

```
git clone https://github.com/gabrielecorbo/oasis-project
```

## How to install the libraries

Install the required libraries listed in the file [`requirements.txt`](./requirements.txt).

# Analysis and results

The repository contains different folders for each of the 6 cities on which the analysis were performed. In each of them are present the different type of datasets needed for the implementation and the scripts to process such data and to make the optimization model.

Moreover, other important files are:

- [`Report_Oasis.pdf`](./Report_Oasis.pdf) contains the description of all the steps of the project and present the obtained results.
- [`webapp.py`](./webapp.py) contains the script to develop a webapp showing the results obtained in each city.

The final presentation can be found here:

- [Final Presentation](./OASIS%20Final%20Presentation.pptx)

# Authors

- Alessandro Amato ([@Amatomato](https://www.github.com/Amatomato))
- Filippo Cipriani ([@SmearyTundra](https://www.github.com/SmearyTundra))
- Gabriele Corbo ([@gabrielecorbo](https://www.github.com/gabrielecorbo))
- Michelangelo Giuffrida ([@michealangel](https://www.github.com/michealangel))
- Riccardo Rosi ([@RickRos98](https://www.github.com/RickRos98))
- Paolo Timis
- Aurelio Zizzo





