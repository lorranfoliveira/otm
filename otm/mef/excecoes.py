class ErroMEF(Exception):
    """Erro que ocorre em problemas do método dos elementos finitos"""
    def __init__(self, erro):
        self.erro = erro

    def __str__(self):
        return self.erro
