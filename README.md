# CPE 593 Final Project

## Neural Network Application in Traffic Management

### Team Members:

Syed Ahmad Shah  
Christina Cuneo

### Project Topic Overview:

Traffic management is the process of monitoring, coordinating, and controlling the movement of vehicles, pedestrians, and other road users so that transportation networks operate safely and efficiently. In urban environments, this is especially challenging because traffic patterns are constantly changing due to congestion, rush-hour demand, turning movements, pedestrian crossings, public transit activity, and the presence of high-occupancy or emergency vehicles. One of the most important tools in traffic management is traffic signal control, since the timing of lights, including protected turn signals, directly affects vehicle flow, waiting times, intersection safety, and the prevention of bottlenecks or gridlock.   
Traditional traffic control methods historically relied on fixed timing schedules or limited rule-based adjustments, which may not respond well to rapidly changing conditions. Because of this, intelligent approaches such as neural networks have become increasingly relevant. By implementing modern tools like neural networks, a model can learn patterns from data and make decisions or predictions based on those patterns in real time. In the context of traffic management, neural networks can be used to analyze traffic conditions and support more adaptive, data-driven control of traffic signals. This can be especially valuable during unusual or unexpected disruptions to typical traffic behavior. This project will examine current approaches to these systems and simulate an urban environment in order to better understand how adaptive models can respond to varying traffic conditions. Through simulation, it becomes possible to evaluate the potential of neural networks to improve efficiency, reduce congestion, and support smarter traffic signal control in complex urban settings.

### Project Objectives:

* Develop a simulation of several traffic intersections in a city-like environment with variable traffic flow and congestion.   
* Utilize a neural network model to determine optimal traffic signal timings  
* Construct a list of flags, variables, and decision trees, for model training and evaluation.  
* Evaluate system performance using metrics:  
  * Average vehicle wait time  
  * Traffic throughput  
  * Queue length per lane  
* Compare results against traditional methods for traffic signal systems, as well as existing research into Neural Network applications within this topic.

### System Design Overview:

1. Traffic Simulation Module  
   1. We will initially test with numerical representations of traffic intersections  
   2. Advance towards utilizing a graphical representation for project demonstration  
   3. Generate real-time state data for decision making  
2. Neural Network Model  
   1. Input: traffic conditions (vehicle counts, wait times, lane congestion, current traffic signals)  
   2. Output: traffic signal decision update (along with duration)  
3. Evaluation Metrics:  
   1. Track performance metrics  
   2. Compare neural network with baseline systems

### Division of Labor:

**Syed Ahmad Shah:**

1. Research and design Neural Network Architecture  
2. Implementation and training of neural network model  
3. Data preprocessing and feature selection

**Christina Cuneo:**

1. Design and implement traffic simulation environment  
2. Research and implement baseline traffic control systems  
3. Implementation of performance metrics and evaluation  
4. Assist in research and implementation of a different neural network model for comparison

### Github Repository Setup:

`finalProject:`

- [`README.md`](http://README.md)  
- `doc/`  
- `assets/`  
- `code/`  
  - `Baseline architecture`  
  - `Neural Network`  
    - `Various neural network architectures`  
- `references/`

[`README.md`](http://README.md)`: Project description and setup instructions`  
`/doc: Current project progress, findings, final paper, presentation`  
`/assets: Project assets, images, etc`  
`/code: Project source code`  
`/reference: Research papers and reference materials`

[Project Repository](https://github.com/AhmadShah-1/Neural-Network-Application-in-Traffic-Management-CPE-593-WS-)
