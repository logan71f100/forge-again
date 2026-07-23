function reloadAfterRestart(x){
  var wasDown = false;
  setTimeout(function poll(){
    fetch('/?_=' + Date.now(), {cache:'no-store'})
      .then(function(r){
        if(!r.ok){ wasDown = true; setTimeout(poll, 800); return; }
        if(wasDown){ window.location.reload(); } else { setTimeout(poll, 800); }
      })
      .catch(function(){ wasDown = true; setTimeout(poll, 800); });
  }, 1200);
  return x;
}
