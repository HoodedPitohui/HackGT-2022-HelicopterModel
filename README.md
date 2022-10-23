# HackGT-2022-HelicopterModel
<p>Akshath Anna, Brian Goldblatt, Karthik Shaji</p>

A challenge for helicopter pilots involves deciding when it’s safe to fly. We developed a python program that provides live visualizations of weather conditions and generates a path of travel.

---

<h3>Libraries</h3>
<ul>
  <li>matplotlib</li>
  <li>numpy</li>
  <li>scipy</li>
  <li>sklearn</li>
  <li>geopandas</li>
  <li>pandas</li>
</ul>

# Inspiration
In an AIAA magazine discussing Advanced Aerial Mobility, one major concern discussed was the lack of collective data on weather for altitude regimes commonly flown by helicopters. We decided to develop our visualizations and algorithms to help current helicopter pilots while keeping the future in mind.

# What it does
We generate multiple visualizations of data that would be of interest to a helicopter pilot, such as dewpoint temperature, air temperature, and weather conditions over an area. Furthermore, we use web scraping off of Aviation Weather (a .gov website), to provide live conditions. Lastly, from a safety metric developed using a logistic regression model, we use an A* algorithm to propose the safest and most distance-efficient path of travel.

# How we built it
We first built an overall project plan with potential goals and steps that we would need to undertake to achieve those goals. Next, we really began to play with the data in every form, and try to identify relevant parameters (such as wind-speed), that could be useful to our model. From this data manipulation, we acquired a better intuition of what might actually be useful for pilots, and then from there used it to develop our safety metrics and proposed path of travel.

# Challenges we ran into
Developing several of these algorithms (such as A*) was very tough, and we also went through a lot of different types of models to create our safety metric (at least 6 hours was spent on that alone).

# Accomplishments that we're proud of
We are proud that we are able to provide an easily-usable, live-updated, weather dashboard that provides pilots a potential path of travel from one latitude/longitude location to another. With this system, we believe they have the tools in their hands to make smarter decisions.

# What we learned
We learned a lot about setting up and passing models through various systems to validate and ensure they work. We also learned about how to implement the path-planning algorithm A, and how to enable certain algorithms to work, even under conditions of sparse data. Lastly, we did a lot of work with the geopandas Python library, which none of us had really worked with before.

# What's next for Helicopter Altitude Weather Data Analyzer
We want to refine the model that we use for the safety metric, as we still believe there’s room for growth in how holistically it takes into account various factors. If were able to potentially obtain more money, we would be interested in searching for datasets that potentially had more information, but were under paywall (which we found several). Lastly, we hope to more accurately account for wind direction when making the A algorithm prediction.
