

def add_events(epochs, behavior, event_type, events):

    if event_type == 'condition':
        epochs.event_id = {'incongruent': 1, 'congruent': 0}
        events = behavior.trial_type.astype('category').cat.codes
    elif event_type == 'laterality':
        drop_ix = behavior.target_location == 'middle'
        behavior = behavior[~drop_ix]
        epochs.drop(drop_ix)
        epochs.event_id = {'incongruent-left': 0,
                           'incongruent-right': 1,
                           'congruent-left': 2,
                           'congruent-right': 3}
        events = []
        for tt, loc in zip(behavior.trial_type, behavior.target_location):
            if tt == 'incongruent' and loc == 'left':
                events.append(0)
            elif tt == 'incongruent' and loc == 'right':
                events.append(1)
            elif tt == 'congruent' and loc == 'left':
                events.append(2)
            else:
                events.append(3)

    else:
        raise ValueError('Unknown Event Type')

    epochs.events[:, -1] = events
    return epochs
