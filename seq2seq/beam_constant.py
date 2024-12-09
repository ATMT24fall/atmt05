import torch

from itertools import count
from queue import PriorityQueue


class BeamSearch(object):
    """ Defines a beam search object for a single input sentence. """
    def __init__(self, beam_size, max_len, pad):

        self.beam_size = beam_size
        self.max_len = max_len
        self.pad = pad

        self.nodes = PriorityQueue() # beams to be expanded
        self.final = PriorityQueue() # beams that ended in EOS

        self._counter = count() # for correct ordering of nodes with same score

    def add(self, score, node):
        """ Adds a new beam search node to the queue of current nodes """
        self.nodes.put((score, next(self._counter), node))

    def add_final(self, score, node):
        """ Adds a beam search path that ended in EOS (= finished sentence) """
        self.final.put((score, next(self._counter), node))

    def get_current_beams(self):
        """ Returns beam_size current nodes with the lowest negative log probability """
        nodes = []
        while not self.nodes.empty() and len(nodes) < self.beam_size:
            node = self.nodes.get()
            nodes.append((node[0], node[2]))
        return nodes

    def get_best(self):
        """ Returns final node with the lowest negative log probability """
        merged = PriorityQueue()
        for _ in range(self.final.qsize()):
            node = self.final.get()
            merged.put(node)

        for _ in range(self.nodes.qsize()):
            node = self.nodes.get()
            merged.put(node)

        best_node = merged.get()
        node = best_node[2]
        if len(node.sequence) < self.max_len:
            missing = self.max_len - len(node.sequence)
            node.sequence = torch.cat((node.sequence, torch.tensor([self.pad]*missing).long()))
        return (best_node[0], node)

    def prune(self):
        """ 
        Compares all nodes and keeps only the beam_size best ones,
        ensuring completed nodes go back to final queue if they rank high enough.
        """
        # Create a new priority queue for merged nodes
        merged = PriorityQueue()
        
        # Track which nodes were from final queue
        final_nodes = set()
        
        # Move all nodes from both queues to merged queue
        while not self.nodes.empty():
            node = self.nodes.get()
            merged.put(node)
        
        while not self.final.empty():
            node = self.final.get()
            # Mark this node as coming from final queue
            final_nodes.add(id(node[2]))  # Use object id as identifier
            merged.put(node)
        
        # Create new priority queues
        new_nodes = PriorityQueue()
        new_final = PriorityQueue()
        
        # Keep only beam_size best nodes
        kept_count = 0
        while not merged.empty() and kept_count < self.beam_size:
            node = merged.get()
            kept_count += 1
            
            # If node was originally in final queue, put it back there
            if id(node[2]) in final_nodes:
                new_final.put(node)
            # # If node ends with EOS but wasn't in final before, add it to final
            # elif node[2].sequence[node[2].length-1] == node[2].search.eos_idx:
            #     new_final.put(node)
            else:
                new_nodes.put(node)
        
        # Update both queues
        self.nodes = new_nodes
        self.final = new_final

    def pad_sequence(self, node):
        """Pads the sequence to max_len while keeping the original sequence in node"""
        # Store original sequence
        node.original_sequence = node.sequence.clone()
        # Pad sequence to max_len
        missing = self.max_len - node.length
        node.sequence = torch.cat((node.sequence.cpu(), torch.tensor([self.pad]*missing).long()))
        return node


class BeamSearchNode(object):
    """ Defines a search node and stores values important for computation of beam search path"""
    def __init__(self, search, emb, lstm_out, final_hidden, final_cell, mask, sequence, logProb, length):
        # Attributes needed for computation of decoder states
        self.original_sequence = sequence  # Store original sequence
        self.length = length  # Actual sequence length
        
        # Pad sequence during initialization
        missing = search.max_len - length
        self.sequence = torch.cat((sequence.cpu(), torch.tensor([search.pad]*missing).long()))
        
        self.emb = emb
        self.lstm_out = lstm_out
        self.final_hidden = final_hidden
        self.final_cell = final_cell
        self.mask = mask
        self.logp = logProb
        self.search = search

    def get_sequence(self):
        """Return sequence for decoder (only up to actual length)"""
        return self.sequence[:self.length]  # Only return sequence up to actual length

    def eval(self, alpha=0.0):
        """ Returns score of sequence up to this node """
        normalizer = (5 + self.length)**alpha / (5 + 1)**alpha
        return self.logp / normalizer

    def get_padded_sequence(self, pad_length):
        """Return sequence padded to specified length"""
        seq = self.sequence[:self.length]  # Get actual sequence
        if len(seq) < pad_length:
            missing = pad_length - len(seq)
            seq = torch.cat((seq, torch.tensor([self.search.pad]*missing).long()))
        return seq
        