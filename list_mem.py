'''
Class that allows an agent to maintain a memory of recently seen artifacts.
'''
class ListMemory():
    '''Simple list memory which stores all seen artifacts as is into a list.
    '''
    def __init__(self, capacity):
        '''
        :param int capacity: The maximum number of artifacts in the memory.
        '''
        self._capacity = capacity
        self._artifacts = []

    @property
    def capacity(self):
        '''The maximum number of artifacts in the memory.
        '''
        return self._capacity

    @property
    def artifacts(self):
        '''The artifacts currently in the memory.
        '''
        return self._artifacts

    def memorize(self, artifact):
        '''Memorize an artifact into the memory.

        If the artifact is already in the memory, does nothing. If memory
        is full and a new artifact is memorized, forgets the oldest artifact.

        :param artifact: Artifact to be learned.
        :type artifact: :class:`~creamas.core.artifact.Artifact`
        '''
        if artifact in self._artifacts:
            return
    
        self._artifacts.insert(0, artifact)
        if len(self._artifacts) > self.capacity:
            self._artifacts = self._artifacts[:self.capacity]