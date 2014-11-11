Dataset
=======

Description
-----------

Group-specific dataset consists of light curves of three different types of eclipsing binaries. 
Light curve is a curve representing brightness of star or star system plotted against time. 

![Eclipsing binary light curve](https://mlnl.net/jg/peripheria/ebs/ex1.png)

Characteristic curve shapes correspond to different types of objects. 
Eclipsing binary stars are star systems where each star periodically eclipse the other. 
Depending on the distance between the stars the system can fall into one of three categories:

- detached
- semi-detached
- contact

![Eclipsing binaries categories](http://lifeng.lamost.org/courses/astrotoday/CHAISSON/AT320/IMAGES/AT20FG21.JPG)

[Further reading](http://books.google.cl/books/about/Eclipsing_Binary_Stars.html?id=W-mVhBx48GwC&redir_esc=y)

[Poster](https://www.tarleton.edu/observatory/posters/Katherine-Poster.pdf)

Data acquisition
----------------

Data is acquired from ASAS catalogue of variable stars ([link](http://www.astrouw.edu.pl/asas/?page=main)). Light curves are downloaded in bulk from [download site](http://www.astrouw.edu.pl/asas/?page=download). Light curve descriptions (with categories) are obtained by scraping from search site displaying [all results](http://www.astrouw.edu.pl/asas/?page=show&qty=all). There is probably a better way but well, this one works too.

After download data from html file is converted into an easier to handle json format and available in zipped form in repository.

Data reduction
--------------

Amount of points and their temporal placement vary greatly so that in order to be able to feed the into the neural network data has to be properly reduced. Extracting curve descriptors for light curve can be done in variety of ways:

- using polynomial chain of piecewise smooth n-th order polynomials (http://arxiv.org/abs/0807.1724 http://arxiv.org/abs/1407.0443)
- using Fourier shape descriptors (http://arxiv.org/abs/0906.0304, http://arxiv.org/abs/0711.0703)


[Spline functions description](http://folk.uio.no/in329/nchap5.pdf)
