# ComputerVision

### Introduction

A repository that is used to store the assignments from the course: ***Computer Vision***, which mostly includes code implementation for some specific questions.  

### Repository Architecture

1. The directory ``CV_Assignment_1`` contains three assignments, including the self-implemented convolution operation, the self-implemented global histogram equalization and the self-implemented local histogram equalization.

    ```
    CV_Assignment_1:
        - assignment_1_q2: convolution operation
            - CODE
        - assignment_1_q3_1: global histogram equalization
            - CODE
        - assignment_1_q3_2: local histogram equalization
            - CODE
        - resource
            - SOURCE IMAGE
        - results
            - PROCESSED IMAGE
    ```

2. The directory ``CV_Assignment_2`` contains two main assignments, each of which contains three sub-questions. The first main assignment requires students to implement 2nd derivative of Gauss, DoG(Difference of Gaussian) with different coefficients and the comparison between the two above. The other one asks students to complete a program that can generate a Gaussian Pyramid. Besides, students are also required to use the DoG function to achieve the functionality of an edge detector. At last, an edge detector with Canny operator needs to be implemented.

    ```
    CV_Assignment_2:
        - assignment_2_q2_1: 2nd derivative of Gauss
            - CODE
        - assignment_2_q2_2: DoG (Difference of Gaussian)
            - CODE
        - assignment_2_q2_3: Comparison between 2nd derivative and DoG
            - CODE
        - assignment_2_q3_1: Gaussian Pyramid
            - CODE
        - assignment_2_q3_2: Edge detector with DoG
            - CODE
        - assignment_2_q3_3: Edge detector with Canny operator
            - CODE
        - resource
            - SOURCE IMAGE
        - results
            - GENERATED IMAGE
    ```

3. The directory ``CV_Assignment_3`` contains one main assignment, which requires student to implement sereval pattern-matching algorithms, from the most basic one to the most common one, including manually selection & calculation, ``SIFT`` and manually calculation with ``RANSAC``. (Due to no available ``SIFT`` modules in ``python3.7`` with ``opencv-python-contrib`` that beyonds version ``3.5.3``, I have to implement the functions in ``MATLAB``)

    ```
    CV_Assignment_3:
        - assignment_3_q1_q2: Pattern-matching functions, including manually selection & calculation, 
        SIFT and RANSAC
        - resource
            - SOURCE IMAGE
        - results
            - GENERATED IMAGE
    ```

### Usage

For more details, please refer to the ``README.md`` in each directory.
