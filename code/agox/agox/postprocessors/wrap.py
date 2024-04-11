from agox.postprocessors.ABC_postprocess import PostprocessBaseClass

class WrapperPostprocess(PostprocessBaseClass):

    name = 'WrapperTheRapper'

    @PostprocessBaseClass.immunity_decorator
    def postprocess(self, candidate):
        candidate.wrap()
        return candidate
        