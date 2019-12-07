Please note: This repository only contains the code, not the data!

This work is based on the ARAGA autoencoder
To be able to run the citation recommendation, you need to download 3 things:
1) Doc2vec model
2) AAN dataset
3) DBLP dataset


<h1>Doc2vec model</h1>
<p>The doc2vec model is necessary to process the abstract and the title from the papers.</p>
<p>You can find it from the KOM server at:</p>
<p> /home/david/Desktop/BA/Datasets/apnews_dbow.bin </p>
<p> /home/david/Desktop/BA/Datasets/apnews_dbow.bin.syn0.npy</p>
<p> /home/david/Desktop/BA/Datasets/apnews_dbow.bin.syn1.neg.npy</p>
<p>Those files have to be copied to</p>
<p> ~/Datasets/apnews_dbow.bin </p>
<p> ~/Datasets/apnews_dbow.bin.syn0.npy</p>
<p> ~/Datasets/apnews_dbow.bin.syn1.neg.npy</p>


<h1>AAN dataset</h1>
<p>The AAN dataset is one of the two datasets that is used for this citation recommendation model.</p>
<p>Please go to (scroll down and fill in the form correctly):</p>
<p> http://tangra.cs.yale.edu/newaan/index.php/home/download (~387 MB)</p>
<p>The only important files are:</p>
<p> aanrelease2014/aan/papers_text/</p>
<p> aanrelease2014/aan/release/2013/acl.txt</p>
<p> aanrelease2014/aan/release/2013/acl-metadata.txt</p>
<p> aanrelease2014/aan/release/2013/author-citation_network.txt</p>
<p> aanrelease2014/aan/release/2013/author_ids.txt</p>
<h3>Copy them to:</p>
<p> ~/Datasets/AAN/raw/papers_text/</p>
<p> ~/Datasets/AAN/raw/acl.txt</p>
<p> ~/Datasets/AAN/raw/acl-metadata.txt</p>
<p> ~/Datasets/AAN/raw/author-citation_network.txt</p>
<p> ~/Datasets/AAN/raw/author_ids.txt</p>


<h1>DBLP dataset</h1>
<p>The DBLP dataset is the second dataset that is used for this citation recommendation model.</p>
<p>Please to to:</p>
<p> https://aminer.org/citation</p>
<p>and download the Version 4 (~193MB).</p>
<p>Extract it and place the file "DBLP-Only-Citation-Oct-19.txt" at</p>
<p> ~/Datasets/DBLP/DBLPOnlyCitationOct19.txt</p>
<p>Now you can run the "run.py" script. It will format the "DBLPOnlyCitationOct19.txt" dataset and dump the files in "raw/" just like the AAN set.</p>


<h1>Aditional</h1>
<p>You need to create some soft links in the directories:</p>
<p> ~/arga2/arga               ===> ~/arga/</p>
<p> ~/output_combination/arga  ===> ~/arga/</p>
<p>This is because some python scripts in "arga2" and "output_combination" access the code in "arga".</p>
<p>If no soft link for the directory is provided, it will not be able to find the link.</p>
<p>The commands are as follows:</p>
<p> ln -s /<absolute-path>/arga arga2/arga</p>
<p> ln -s /<absolute-path>/arga output_combination/arga</p>

<h3>The build up should now be finished.</h3>
<h3>You can now start to run the model. E.g.:</h3>

<p>cd Datasets/</p>
<p>nano preprocess.py # set the dataset: AAN/DBLP and select the features to be taken</p>
<p>python2.7 preprocess.py # let the script process the data</p>
<p>cd ..</p>
<p>./copy AAN # copy the files from the "Dataset" directory into the "data" directory</p>
<p>./copy DBLP</p>
<p>cd arga # or cd arga2</p>
<p>nano run.py # select the dataset and the features once again</p>
<p>nano settings.py # set the settings you want</p>
<p>python2.7 run.py # let the model run. This can take multiple minutes. The performance will be shown at the end.</p>
<h3>Optional</h3>
<p>cd ..</p>
<p>cd output_combination/</p>
<p>nano output_combination # select the features, the history and other collaborative features to be taken in the filtering step</p>
<p>python2.7 output_combination.py # run the filtering step. The results will be shown at the end.</p>


<h3>Congratulaitons, you successfully run the model.</h3>
 


