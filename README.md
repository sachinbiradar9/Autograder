# Autograder
:trophy: Won the NLP Project Competition with test accuracy of 94%  

System that can automatically grade student essays. NLP techniques were the center of automated scoring. Various factors to grade were -

<ol type="a">
  <li>Length of the essay</li>  
  <li>Spelling mistakes</li>
  
  <li>Syntax/Grammar</li>
  <ol type="i">
    <li>Subject-Verb agreement</li>
    <li>Verb tense / missing verb / extra verb</li>  
    <li>Sentence formation</li>
  </ol>
  
  <li>Semantics</li>
    <ol type="i">
      <li> Text coherent</li>
      <li>Topic coherence</li>
    </ol>
</ol>

The final score is calculated using the below formula -  
Final Score = ```2a − 2b + 0.2c.i + 0.8c.ii + 2c.iii + 2d.i + 3d.ii```

## Installation
Just run the bash file as mentioned below and the installlation will be done automatically

## Usage
Make sure [stanford core nlp](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip) server is running.  
Go to executable folder and run  
`bash run.sh`

## Credits
- [Dr. Barbara Di Eugenio](https://www.cs.uic.edu/k-teacher/barbara-di-eugenio-phd/)
- Abhinav Kumar
