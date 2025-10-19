---
name: Feature request
about: Suggest an idea for this project
title: "Add Dynamic Greeting and Motivational Footer"
labels: enhancement, triage
assignees: AstraBert
---

**Is your feature request related to a problem? Please describe.**  
I'm always frustrated when the app finishes processing a document and there’s no visual or emotional feedback for the user. The interface feels functional but a bit robotic, especially after long operations. A small personal touch could make the experience more engaging and human.

**Describe the solution you'd like**  
I’d like to add a **dynamic greeting and motivational footer** at the bottom of the Streamlit interface.  
The greeting changes automatically based on the time of day (morning, afternoon, evening), and a random motivational quote is displayed each time the page loads or a document is processed.  

This would make the app feel friendlier and more alive while giving users a small dopamine boost after completing tasks.

**Describe alternatives you've considered**  
- Showing a random fun fact or coding tip instead of a quote.  
- Displaying user-specific greetings fetched from their account name.  
- Including a small animated emoji or confetti effect along with the greeting for extra engagement.  

**Additional context**  
Here’s an example of what the implementation could look like:

```python
import datetime
import random
import streamlit as st

current_hour = datetime.datetime.now().hour
if current_hour < 12:
    greeting = "Good morning ☀️"
elif current_hour < 18:
    greeting = "Good afternoon 🌤️"
else:
    greeting = "Good evening 🌙"

footer_quotes = [
    "“Simplicity is the soul of efficiency.” – Austin Freeman",
    "“Code is like humor. When you have to explain it, it’s bad.” – Cory House",
    "“Programs must be written for
