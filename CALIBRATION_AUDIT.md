# Validação da redução CCD — 18 outubro 2026

A implementação anterior não seguia integralmente o guia: combinava sem rejeição de valores extremos e dividia a ciência por contagens de flat não normalizadas. Essa divisão dependia da iluminação dos flats e alterava a escala da ciência. A receita foi corrigida; os resultados antigos devem ser recalculados.

## Receita implementada

1. **Master bias:** média por píxel, com rejeição bilateral a 5σ, centro na mediana e dispersão MAD convertida para σ (`astropy.stats.sigma_clip`, até cinco iterações). Não se alinham nem normalizam bias.
2. **Master flat:** para cada filtro/exposição selecionado, subtrair o master bias de cada flat; propagar máscaras e excluir resposta não positiva; dividir cada imagem pela sua mediana válida; combinar com média e rejeição a 5σ; normalizar o resultado para mediana 1. Não alinhar flats.
3. **Ciência individual:** `S_corr = (S_original − B_master) / (F_master / mediana_válida(F_master))`. O denominador é adimensional. Os valores científicos conservam a unidade de contagens, ADU quando o cabeçalho não especifica BUNIT. Valores negativos de ciência são preservados; respostas de flat inválidas são mascaradas.
4. **Alinhamento:** depois da correção, em coordenadas do céu, com interpolação uma única vez. As bordas sem cobertura ficam mascaradas. Defeitos isolados são excluídos da medição de deslocamentos, sem inventar píxeis científicos.
5. **Resultado final:** combinar apenas imagens corrigidas e alinhadas do mesmo objeto, filtro e exposição, com média e rejeição a 5σ. Grupos com menos de duas imagens não são combinados; a imagem individual corrigida continua disponível para download.
6. **Exportação:** FITS mantém valores de ponto flutuante e máscara. PNG é uma representação visual com quantização; usar FITS para medições científicas.

A mediana continua disponível como alternativa explícita. Os cabeçalhos registam COMBMETH, CLIPSIG, CLIPCEN, CLIPDEV, NCOMBINE, BIASFILE, FLATFILE, FLATNORM e HISTORY. A seleção de ficheiros, os métodos e as calibrações invalidam resultados anteriores. A atualização da receita também invalida resultados em memória.

A opção `-bias` nos flats mostra agora a resposta calibrada e normalizada. Desativada, mostra a combinação crua apenas para inspeção. As escalas visuais são separadas porque as unidades são diferentes. O auxiliar legado `correct_flat_groups` é apenas uma subtração diagnóstica e deixou de preparar os flats do fluxo científico.

## Inspeção dos ficheiros disponíveis

Pesquisa recursiva no projeto, incluindo ficheiros ignorados/ocultos e excluindo apenas o ambiente Python e metadados Git: **299 FITS**, todos em `share-20261001`.

| Tipo no cabeçalho | Quantidade |
|---|---:|
| BIAS | 21 |
| FLAT | 116 |
| OBJECT | 162 |
| DARK | 0 |

A temperatura CCD registada varia entre −5,15 e −4,88 °C. Nenhum dos 299 ficheiros identifica regiões BIASSEC, TRIMSEC ou DATASEC, nem as variantes de overscan pesquisadas. Não houve erros de leitura dos cabeçalhos. Isto documenta os dados do projeto; não representa uma pesquisa de todos os discos do computador.

## Limitações que os ficheiros não resolvem

- **Darks:** não estão disponíveis. O programa não subtrai darks neste fluxo e regista DARKCOR=False. A ausência de darks não demonstra que a corrente escura, amp glow ou píxeis quentes sejam desprezáveis. Para certificar essa parte são necessários darks compatíveis ou caracterização da câmara. Se darks forem usados no futuro, retirar bias antes de os escalar e evitar subtrair bias duas vezes.
- **Overscan:** a ausência de palavras-chave não prova que não exista uma região não iluminada. Não se escolhe nem corta uma região arbitrária. É preciso confirmar o modelo/formato da câmara; o próprio guia alerta que até BIASSEC pode estar errado.
- **Qualidade de aquisição:** os cabeçalhos não certificam linearidade/saturação, iluminação uniforme dos flats, ausência de estrelas nos flats, estabilidade ótica ou adequação para fotometria. O alinhamento implementa translação, não correção de distorção/rotação de campo.
- **Máscaras e incerteza:** máscaras existentes, não finitos, resposta não positiva e bordas são propagados. Não se produz automaticamente uma máscara física de hot pixels sem darks, nem uma imagem de variância para incertezas fotométricas. Rejeição com poucas imagens tem eficácia limitada.

Estas limitações impedem afirmar que os dados estão integralmente certificados para fotometria, mesmo quando a receita bias/flat está matematicamente correta.

## Verificação executada

- 89 testes existentes passaram após atualização das expectativas para a receita normalizada; mais 4 regressões novas passaram com bias contaminado, iluminação variável de flats, resposta inválida, conservação da escala científica e combinação com rejeição por exposição e exportação de flats normalizados.
- Processamento dos dados reais em resolução integral (1536 × 1024): 21 bias, todos os 116 flats candidatos e correção das 162 imagens de ciência. Cada imagem corrigida foi comparada numericamente com a fórmula independente. A execução demorou cerca de 28 segundos fora da interface.
- Master bias: mediana de 543,095238 ADU. Sete masters flats normalizados com mediana 1; 111 flats contribuíram. Cinco flats pertencem a grupos de filtro/exposição sem pelo menos duas imagens e não foram combinados. Nenhum píxel foi mascarado por resposta inválida nesta amostra; isso não certifica ausência de defeitos físicos.
- Os 162 science distribuem-se por 18 grupos de objeto/filtro/exposição; dois grupos têm apenas uma imagem. As correções individuais existem para todos, mas a combinação exige pelo menos duas imagens.
- Os testes não alteraram os ficheiros originais. Os produtos reais de validação ficaram em memória/ficheiros temporários, não foram publicados nem substituíram resultados do utilizador.

Amostra real final: quatro imagens de HD226868, Blue, 30 s, corrigidas, alinhadas e combinadas em resolução integral. O resultado conserva ADU e NCOMBINE=4; os deslocamentos medidos ficaram entre 0 e −4,34 píxeis em Y e entre +1,23 e −28,93 em X.

A preparação completa dos flats e dos dois ZIPs ultrapassou o antigo limite de 30 segundos. O limite desta operação foi aumentado para 180 segundos em Combine, auto e na preparação feita por Correct. A repetição através do mesmo processo isolado da interface terminou com sucesso em 30,59 segundos, produzindo sete flats normalizados e ambos os ZIPs.

## Referências consultadas

- [Prefácio e âmbito do guia](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/00-00-Preface.html).
- [Combinação de imagens](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-06-Image-combination.html): média com rejeição robusta.
- [Master bias](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/02-04-Combine-bias-images-to-make-master.html): exemplo com limiares 5σ e MAD.
- [Calibração do bias e overscan](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/02-02-Calibrating-bias-images.html).
- [Conceitos de flat field](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-01-about-flat-corrections.html) e [calibração dos flats](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-03-Calibrating-the-flats.html).
- [Combinação dos flats](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-04-Combining-flats.html): normalização individual pela mediana antes de combinar.
- [Calibração da ciência](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/06-03-science-images-calibration-examples.html).
- [Escolhas de calibração](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-09-Calibration-choices-you-need-to-make.html) e [máscaras](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/08-05-incorporating-masks-into-calibrated-science-images.html).

A edição dev contém erros de execução e lapsos nos exemplos; a implementação segue os princípios descritos, não copia cegamente as células. A normalização usa a mediana válida explicitamente; `ccdproc.flat_correct` usa a média por defeito, mas permite escolher o valor de normalização. A escolha da mediana está documentada e testada.
