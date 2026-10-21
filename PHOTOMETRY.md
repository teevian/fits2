# Análise astrofísica observacional

Na secção Science, depois de criar os masters, o botão **Análise astrofísica**
(junto dos resultados de Generate) abre o popup de análise com gráficos brancos.
A imagem RGB é um produto de visualização. Os cálculos usam apenas os FITS lineares
por filtro ou as exposições corrigidas individuais disponíveis na sessão.

## Procedimento

1. Selecionar os FITS e, em **Selecionar estrelas**, escolher **Alvo** ou
   **Referência** e clicar na estrela na imagem. Há zoom até 400%, círculos de
   abertura/fundo e ajuste manual de x/y (origem zero; y cresce para baixo).
   O alvo inicial é apenas o píxel mais brilhante. A referência começa vazia.
   Não há associação automática nem cópia de coordenadas entre filtros.
2. Escolher abertura e anel de fundo, evitando vizinhos, asas da PSF e gradientes.
   O mesmo raio em píxeis não garante a mesma fração de fluxo em filtros distintos.
   Verificar a curva de crescimento e fornecer correções de abertura calibradas.
3. No modo predefinido **Referência no mesmo FOV**, selecionar uma estrela de
   comparação estável e isolada e indicar a sua magnitude tabelada **por filtro**,
   o erro 1σ (vazio = desconhecido), identificação/catálogo e banda/sistema.
   O algoritmo mede ambas as estrelas na mesma imagem e com os mesmos raios,
   calcula a razão de fluxos e o ponto zero da referência.
4. Introduzir ganho e limite de linearidade/saturação quando conhecidos. O ganho
   não é deduzido de um cabeçalho que possa conter apenas uma configuração do ADC.
   O limite deve ser válido na escala dos dados corrigidos usados. A saturação de
   exposições originais pode ficar oculta num master e requer inspeção dos inputs.
5. Como alternativa, no modo **Ponto zero manual**, ativar `calibrated` e fornecer
   `Z_eff`, `sigma_Z` e a
   referência/sistema da calibração. `Z_eff = Z − kX + cK + correção de abertura`.
   Os coeficientes devem ser obtidos de padrões/comparações; não são ajustados
   automaticamente. Erros de magnitude padrão assumem independência entre a
   medição e a calibração. A aplicação não converte ADU em fluxo energético.
6. Calcular e verificar mapas, tabelas, perfis, resíduos, crescimento e histogramas.
   Exportar o ZIP com CSVs, parâmetros/cabeçalhos/referências em JSON e HTML autónomo.
   Os gráficos permitem exportação SVG através da barra do Plotly.

## Calibração com uma referência do mesmo campo

### Pesquisa de magnitudes por nome

O campo **Nome da estrela de referência** começa em **V1674 Cyg**. Enter ou
**Pesquisar** submete uma única consulta a SIMBAD e VizieR; sair do campo não
dispara pedidos adicionais. Todos os filtros disponíveis são selecionados
inicialmente, e cada banda identificada recebe a sua própria sugestão. As sugestões
por banda preenchem os campos de magnitude e erro, que continuam editáveis.
Alterações manuais são preservadas durante os restantes ajustes do popup.
Escolher outra estrela, banda ou sugestão aplica os novos valores de catálogo.

A consulta inclui a fotometria SIMBAD e os catálogos VizieR APASS DR9 (II/336),
Johnson UBVRIJHK (II/7A) e médias UBV (II/168). Mostram-se a fonte, a publicação
quando disponível e a data UTC da consulta. SIMBAD e VizieR são dois serviços do
CDS e podem reproduzir a mesma publicação: não constituem confirmação independente.
O relatório exporta a consulta, os valores originais e a indicação de edição manual.

Não se inventam erros ausentes nem se usa dispersão de catálogo como erro da média.
Associações ambíguas não são selecionadas automaticamente. A pesquisa VizieR usa
um raio de 3 segundos de arco; movimento próprio e épocas diferentes podem impedir
uma correspondência. Confirma sempre que o objeto identificado é a estrela clicada.
Filtros RGB genéricos exigem escolha explícita da banda; o preenchimento não valida
a transformação para um sistema padrão. Falhas de consulta permitem entrada manual.

### Cálculo diferencial

`m_T = m_R − 2.5 log10(F_T/F_R) + Δ_corr`. Como os dois fluxos vêm do mesmo FITS,
a exposição comum cancela. A magnitude diferencial é possível mesmo sem EXPTIME;
o ponto zero na convenção ADU/s só é apresentado quando esse tempo é conhecido:
`Z_R = m_R − m_inst,R` (sem a correção diferencial adicional).

A magnitude `m_calibrated` é ancorada na referência. Só alimenta `m_std` e o
painel H–R quando o utilizador confirma banda padrão compatível ou transformação
validada e identifica o sistema. Uma magnitude V aplicada a um filtro RGB genérico
não é automaticamente uma magnitude Johnson V. O campo de correção adicional é
`Δ_corr = −k ΔX + c Δcor + diferença de correção de abertura`; zero pressupõe
termos desprezáveis, não que tenham sido medidos. Comparações semelhantes em cor,
sem variabilidade e com resposta linear são preferíveis.

Para entradas independentes, e com todos os erros disponíveis:
`σ²(m_T) ≈ σ²(m_inst,T) + σ²(m_inst,R) + σ²(m_R,tabelada) + σ²(Δ_corr)`.
Erro de catálogo vazio não é tratado como zero. Em masters, podem ser fornecidos
separadamente erros totais de fluxo do alvo e da referência. Não se calcula um erro
completo quando falta um termo. Aberturas demasiado próximas para os anéis são
rejeitadas; anéis de fundo sobrepostos permitem a magnitude, mas suprimem a
propagação independente sem uma estimativa da covariância. Asas extensas de PSF
e outros vizinhos continuam a exigir inspeção. Não há análise automática de
estabilidade da comparação nem ensemble de várias referências nesta versão.

O relatório inclui coordenadas, identificação, magnitude/erro de catálogo,
correção adicional, fluxo e diagnóstico da referência. Mudar a estrela, magnitude,
banda ou parâmetros invalida os resultados anteriores. Os gráficos da referência
ficam num expansor e também são exportados, juntamente com os seus perfis CSV.

## Fotometria e precisão

A abertura inclui píxeis cujos centros estão dentro do raio: não usa pesos
fracionários. Píxeis inválidos na abertura e anéis fora da imagem são rejeitados.
No anel, excluem-se máscaras e não finitos, aplica-se clipping iterativo 3σ em
relação à mediana/MAD e calcula-se a média e variância amostral dos sobreviventes.
São exigidos pelo menos 20 píxeis de fundo.

`F = sum(C) − n b`, `q = F/t`, `m_inst = −2.5 log10(q / [1 ADU/s])`.
A exposição é a referência EXPTIME/EXPOSURE do FITS, inclusive quando a combinação
normalizou todas as exposições para esse tempo. Não se soma o tempo do stack.
Sem exposição válida ou fluxo positivo, a magnitude não é calculada.

Para uma exposição não interpolada, com ganho g conhecido:
`var(F) ≈ F/g + n (1+n/m) s_b² + sigma_cal²`. O termo de calibração adicional
existe na API (zero por omissão); incertezas totais determinadas externamente
podem ser introduzidas no popup. O ruído de leitura já presente no fundo não
é contado novamente. O erro gaussiano de magnitude só é apresentado para S/N ≥ 5.
Clipping, fundo não uniforme, flat e erros partilhados podem invalidar esta
aproximação; a aplicação explicita esta limitação.

**Masters combinados/interpolados não recebem S/N ou erros inventados.** A imagem
não contém a variância/covariância necessária. Introduzir `sigma_flux` apenas se
for uma estimativa externa validada do erro total de abertura, incluindo fonte,
fundo, covariância e calibração. Este valor substitui, não acrescenta, o modelo
analítico de ruído. A ausência de erro não significa erro zero.

O perfil é a média de anéis de largura 1 pixel. As barras representam dispersão
interna/√N, sem o erro comum do fundo ou covariância entre píxeis. O ajuste Moffat
é descritivo, não ponderado, aos anéis: não se apresenta uma incerteza formal de
FWHM nem se identifica FWHM como seeing. Não se assume uma escala angular a partir
das dimensões contraditórias dos píxeis referidas nas notas do projeto.

## Temperatura, H–R e vida estelar

É preciso confirmar medições da **mesma estrela**, Johnson B e V calibradas em
Vega, distância independente positiva, E(B−V), A_V, BC_V e respetivas referências.
Filtros RGB genéricos não satisfazem automaticamente estas condições.

A temperatura usa a aproximação de Ballesteros (2012), com intervalo operacional
conservador −0.4 ≤ (B−V)₀ ≤ 2.0. Isto não valida a relação para qualquer tipo estelar.
Supergigantes como Deneb, binárias como HD 226868 e fontes peculiares exigem modelos
atmosféricos/evolutivos adequados; não devem receber parâmetros físicos para
publicação apenas através desta aproximação.

`M_V = V − 5 log10(d/10 pc) − A_V`, `M_bol = M_V + BC_V` e
`L/L_sun = 10^[0.4 (4.74 − M_bol)]`. Mostram-se um ponto H–R e um ponto
cor–magnitude por par selecionado; não se fabrica uma população estelar.

A propagação opcional usa derivadas de primeira ordem e entradas independentes.
Requer erros das magnitudes calibradas, permite erros de distância/extinção/BC/massa,
e exclui a dispersão intrínseca da relação de cor/modelos. As barras são marginais;
os eixos partilham V. Extinções e calibrações correlacionadas exigem um modelo
conjunto. Não são calculadas idades, isócronas ou massas a partir de RGB.

Só com classificação independente de sequência principal e massa independente se
apresenta `tau_MS ≈ 10^10 (M/M_sun)/(L/L_sun) anos`. É uma escala aproximada de vida
na sequência principal, não a idade nem o tempo restante, e não é aplicável à fase
atual de uma supergigante ou a um sistema binário não resolvido sem modelação.

## Referências

- [AAVSO: Guide to CCD Photometry, fotometria diferencial](https://www.aavso.org/sites/default/files/CCDPhotometryGuide_3.pdf).

- [Photutils: metodologia de fotometria de abertura](https://photutils.readthedocs.io/en/stable/user_guide/aperture.html)
  (referência metodológica; esta implementação usa NumPy/Astropy/SciPy, sem dependência Photutils).
- [Ballesteros 2012](https://arxiv.org/abs/1201.1809).
- [IAU 2015 B2: escala bolométrica](https://arxiv.org/abs/1510.06262).
- [Princeton: escala de vida na sequência principal](https://www.astro.princeton.edu/ast203/Lectures/Lecture_11.html).

Os testes sintéticos verificam equações, máscaras, saturação, ausência de metadados,
restrições de stacks, exportação, abertura do popup e invalidação de resultados.
Não substituem uma validação observacional com padrões e fontes de controlo.
