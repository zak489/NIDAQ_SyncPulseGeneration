# NIDAQ_SyncPulseGeneration
The repositiory is designed to facillitate synchronize digital (and hopefully analog) pulse generation from a NIDAQ card, for both cases where it uses the internal clock or external trigger to synchronize the output. The pulse synthax would ideally follow that from the Swabian pulse streamer to enable seemless transitions.

## Inspiration
There is already a jupyter notebook on this, which can be found here https://nbviewer.org/gist/baldwint/0c96f2f7bbeb90af4626.
This is done in Python 2.7 and with the older PyDAQMx modules. The goal here is to integrate this functionality to nidaqmx.

## Feedback Welcome
I am well-awared that this is not new, but my difficulty of finding the necessary codes for this, prompts me to write and hopefully publish this to the world, so new user can exploit the full functionality of the NIDAQ cards for analog input, output, digital input, digital output, as well as pulse counting.


