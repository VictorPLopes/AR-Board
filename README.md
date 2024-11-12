# Lousa Virtual

## Descrição

Este projeto é uma lousa virtual, onde o usuário pode desenhar utilizando a webcam e canetões vermelho e verde. O projeto foi desenvolvido utilizando a biblioteca OpenCV e Python para a disciplina de Processamento Digital de Imagens.

![Lousa Virtual](assets/video.gif)

## Utilização

Para rodar o projeto, é necessário ter o Python (3.12 ou mais recente) e a biblioteca OpenCV instalados. Para instalar a biblioteca OpenCV, execute o seguinte comando:

```bash
pip install opencv-python
```

Após baixar/clonar o repositório, execute o arquivo `main.py`:

```bash
python main.py
```

Ao executar o arquivo, a webcam será aberta e o usuário poderá desenhar na tela utilizando os canetões vermelho e verde.

A tela da webcam mostra o que está sendo detectado em tempo real (a ponta dos canetões) e o desenho que está sendo feito.

É possível ajustar os valores de detecção dos canetões através dos trackbars que aparecem na parte superior da tela.

Para limpar a tela, basta pressionar a tecla `c`.

Para sair do programa, basta pressionar a tecla `q`.

## Autores

- Allan Bastos da Silva
- Mateus Carvalho Lucas
- Victor Probio Lopes
- Wilson Bin Rong Luo
