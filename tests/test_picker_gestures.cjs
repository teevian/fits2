// Run with: node --test tests/test_picker_gestures.cjs
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {readFileSync}=require('node:fs');
const vm=require('node:vm');
const script=readFileSync(require('node:path').join(__dirname,'../science_picker_component/index.html'),'utf8').split('<script>')[1].split('</script>')[0];
function picker(initial=Array.from({length:40},(_,i)=>String(i))) {
  class Element {
    constructor(){this.listeners={};this.style={};this.dataset={};this.children=[];this.attrs={};this._scrollTop=0;this.clientHeight=320;}
    addEventListener(type,fn){(this.listeners[type]??=[]).push(fn)}
    setAttribute(name,value){this.attrs[name]=value}
    append(...children){this.children.push(...children)}
    replaceChildren(){this.children=[]}
    closest(){return this.className==='tile'?this:null}
    contains(node){return this.children.includes(node)}
    querySelectorAll(){return this.children.filter(x=>x.className==='tile')}
    focus(){} setPointerCapture(){}
    getBoundingClientRect(){return {top:0,bottom:320,left:0,right:200}}
    get scrollHeight(){return 1280}
    get scrollTop(){return this._scrollTop}
    set scrollTop(value){this._scrollTop=Math.max(0,Math.min(960,value))}
  }
  const elements=Object.fromEntries(['grid','count','apply','cancel','footer'].map(k=>[k,new Element()]));
  const grid=elements.grid, messages=[], frames=new Map(), window=new Element();
  let frameId=0,eventId=0,clears=0;
  const parent={innerHeight:1000,postMessage:message=>messages.push(message)};
  Object.assign(window,{innerHeight:370,frameElement:new Element(),getSelection:()=>({removeAllRanges(){clears++}})});
  const document={getElementById:id=>elements[id],querySelector:()=>elements.footer,createElement:()=>new Element(),
    elementFromPoint(x,y){if(x<0||x>=200||y<0||y>=320)return null;return grid.children[Math.floor((y+grid.scrollTop)/64)*2+Math.floor(x/100)]}};
  const context=vm.createContext({window,parent,document,crypto:{randomUUID:()=>String(++eventId)},
    requestAnimationFrame:fn=>{frames.set(++frameId,fn);return frameId},cancelAnimationFrame:id=>frames.delete(id)});
  vm.runInContext(script,context);
  window.listeners.message[0]({source:parent,data:{type:'streamlit:render',args:{inline:true,selected:initial,
    items:Array.from({length:40},(_,i)=>({name:String(i),image:'preview.png'}))}}});
  function emit(type,options={}){
    const event={type,button:0,pointerId:1,clientX:50,clientY:20,target:grid.children[0],deltaY:0,deltaMode:0,
      preventDefault(){this.prevented=true},stopPropagation(){this.stopped=true},...options};
    for(const fn of grid.listeners[type]||[])fn(event);
    return event;
  }
  return {grid,emit,selected:i=>grid.children[i].attrs['aria-pressed']==='true',
    values:()=>messages.filter(m=>m.type==='streamlit:setComponentValue').map(m=>m.value),
    tick(){const [id,fn]=frames.entries().next().value;frames.delete(id);fn()},clears:()=>clears};
}
test('text selection is blocked on press, drag and selectstart',()=>{
  const p=picker();assert.ok(p.emit('pointerdown').prevented);assert.equal(p.clears(),1);
  assert.ok(p.emit('mousedown').prevented);assert.ok(p.emit('selectstart').prevented);
  assert.ok(p.emit('pointermove',{clientY:80}).prevented);
});
test('wheel while held paints all crossed rows using the initial deselection operation',()=>{
  const p=picker();p.emit('pointerdown');
  const wheel=p.emit('wheel',{deltaY:256});
  assert.ok(wheel.prevented&&wheel.stopped);assert.equal(p.grid.scrollTop,256);
  for(const i of [0,2,4,6,8])assert.equal(p.selected(i),false);
  for(const i of [1,3,5,7,9])assert.equal(p.selected(i),true);
  assert.equal(p.values().length,0);p.emit('pointerup');
  assert.equal(p.values().length,1);assert.equal(p.values()[0].action,'change');
});
test('starting on an unselected tile selects, and crossing a selected tile never deselects it',()=>{
  const p=picker(['2','6']);p.emit('pointerdown');p.emit('wheel',{deltaY:256});p.emit('pointerup');
  for(const i of [0,2,4,6,8])assert.equal(p.selected(i),true);
});
test('lower and upper edges scroll and paint without further pointer movement',()=>{
  const p=picker();p.emit('pointerdown');p.emit('pointermove',{clientY:318});
  for(let i=0;i<12;i++)p.tick();
  assert.ok(p.grid.scrollTop>100);assert.equal(p.selected(12),false);
  p.emit('pointermove',{clientY:2});const before=p.grid.scrollTop;p.tick();
  assert.ok(p.grid.scrollTop<before);p.emit('pointerup');assert.equal(p.values()[0].action,'change');
});
test('cancelled gestures restore selection, and double-click views without changing selection',()=>{
  const p=picker();p.emit('pointerdown');p.emit('wheel',{deltaY:128});p.emit('pointercancel');
  assert.equal(p.selected(0),true);assert.equal(p.values().length,0);
  for(let i=0;i<2;i++){p.emit('pointerdown');p.emit('pointerup');p.emit('click')}
  p.emit('dblclick');assert.equal(p.selected(0),true);assert.equal(p.values().at(-1).action,'view');
});
