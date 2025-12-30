I have a need to quickly spin up a new project based on the POC that is in this branch.

For quick context, I grabbed haindy and modified it so it can do computer vision and use a software keyboard and mouse to operate at the OS level instead of using playwright and the browser. 

This POC is very successful so now I want to use it as base for a new project. Please read the design in docs/design/linkedin_desktop_poc.md to understand how it works, then read te codebase. 

The idea is to make a task specific agentic system that will be a stripped down version of haindy. Here's what it will do:

1. The agent will run and live on. It will open a browser where there's already a linkedin session. 
2. This agent will wait or listen for linkedin profile URLs in a slack channel
3. Once it gets a linkedin profile, it will start working on compiling the candidate info 
   1. Get contact info by clicking in the contact info modal, then closing. 
   2. Get Skills by scrolling down and expanding, then going back to the profile.
   3. Get Education by scrolling down and expanding, then going back to the profile.
   4. Get Experience by scrolling down and expanding, then going back to the profile.
4. Get the presigned url for the image by clicking the profile picture and in the pop up doing right click, copy image url. Also put this in the json.
5. Make a POST to our backend 
6. Possibly a GET to validate that all fields were correct 
7. Write a response to the slack channel. 


So this data entry agentic system needs the following: 

- A predefined plan which cover the above list. We can repurpose the json test plans for this. We don't really needa test planner agent here. 
- A operator agent that will grab the tasks and break them into computer actions. We can repurpose the test runner agent here with minimal modifications. 
- An action agent that will perform the actions using computer vision and soft keyboard / mouse. This is basically the same action agent we already have. I think we could basically just move files or copy /paste for this. 

Important considerations: 

1. Avoid the DOM as we have already proven with the POC.  
2. Be smart about linkedin in particular. We will long term cache the position of the elements so that we only have to call the computer use model to understand the position of the elements (and how to interact with them) once. 
3. Have a client to our backend, so that we can do the POST to upload the candidate. 
4. Ideally connect to slack where it will get linkedin URLs to scrape. 
5. Contrary to haindy, it will live on, so that we don't have to worry about opening it and shutting it down. 


If you understand this plan I want you to make a plan file for it, in .md like the other ones. 
And I also want you to ask any questions you might have at this time. Ideally before writing the plan, which I'll review once you get it done and provide feedback. 

--------------------------------------

I'll answer in order:

1. The workspace will be the one that is already open, and the channel will be #data-entry-agent-test. The reply will be in the channel as a reaction, probably a checkmark. Same for the ack, it can be a clock icon. 
2. I will provide an example of the json payload, we need to follow the same example. I would put the example in the prompts that create the payload so that it's clear. I'm not sure what the confirmation GET is, leave that TBD. 
3. The json will give you the format and depth. We need the data to fit the json. 
4. This is an interesting queston. Since this is going to be computer use there's really only one set of keyboard and mouse, so I don't see how we could do concurrency without taking this to multiple vm arena. For now it will run just in my computer and we will deal with scaling later. The important thing is, the agent will just run one task at a time. I don't think queue is necessary because if you think about it, the slack window will be open, the agent will look for new candidates in the channel, if there's one that doesn't have the clock emoji reaction and no chekc mark emoji reaction, then it will grab it and switch to the browser window, which will take slack to the background, and never see it again until it completes the job and brings slack forward again. Think of how a human being would do it. 
5. Yes still firefox, single monitor, 1080p, just like this POC. The only extra thing is switching between slack and firefox. 
6. I'd say keep like the last 5 candidates or so, no need to persist anything further than that. And yes, it is okay to cache things in the best way possible and with the most amount of data possible. Ideally we will only take screenshots for the model to see and write down data, and to validate tasks (not individual actions) were actually performed. In an ideal scenario we would take just 9 screenshots: the ones that we need in steps 1 to 7 of the tasks lists that are not curl tool calls, and the rest would be cached. 

----------------------------

I've made this plan: docs/plans/linkedin_profile_agent.md to fork this haindy POC into it's own project / repo, which will be much more lean.

First please read the plan carefully. I've created an empty repository here:
/home/fkeegan/src/teal/teal-data-entry-agent

And I would like you to make this re-implementation / fork into this new repo. 
Browse the current code and understand it. Copy / write only what you need and leave everything else behind.

Before started ask any questions you may have. 

--------------------------------

Let me answer in order:

1. Production endpoint: https://api.teenhire.co/platform/sync-task (POST, JSON body { "data": { ... } }, Content-Type: application/json). No auth is required by the backend. We should include live but also a dry run mode as a start up parameter so we can test before doing the POST.
2. What? This is computer vision and we will be taking screenshots of the entire screen. Slack will be open in a window, already signed in to the correct workspace. 
3. Yes to JSON on disk under data/ and yes to carry over all cache. 
4. Yes I want playwright, web-driving and any other unnecessary code to be removed completely. Well, actually not added. Remember that this is a very specific agent. 
5. Yeah let's keep that. 
6. Yes to all of this point. 

------------------------------------

I think I did not make myself clear about slack. We are not programatically using slack. There's going to be a slack window in the computer and we will do computer use with mouse clicks by coordinate and keyboard typing. All we need is a before task that makes sure to bring slack up front, make sure we are in the right channel and look for new activity in the channel every 5 / 10 minutes. 