import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { AmbientLight, DirectionalLight, HemisphereLight } from 'three';
@Component({
  selector: 'app-model-viewer',
  template: '<div class="model-container" #modelContainer></div>',
  styleUrls: ['./model-viewer.component.scss']
})
export class ModelViewerComponent implements OnInit {
  @ViewChild('modelContainer', { static: true }) modelContainer: ElementRef;

  ngOnInit() {
    this.initScene();
  }

private initScene() {
  const scene = new THREE.Scene();
  
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer();
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  this.modelContainer.nativeElement.appendChild(renderer.domElement);
  renderer.setClearColor(0x171717);

  // Add ambient light to the scene
  const ambientLight = new AmbientLight(0xffffff, 0.9);
  scene.add(ambientLight);

  // Add directional light to the scene
  const directionalLight = new DirectionalLight(0xffffff, 1);
  directionalLight.position.set(1, 1, 1);
  scene.add(directionalLight);

  // Add hemisphere light to the scene (for softer global illumination)
  const hemisphereLight = new HemisphereLight(0xffffff, 0x000000, 0.4);
  scene.add(hemisphereLight);

  const loader = new GLTFLoader();
  const controls = new OrbitControls(camera, renderer.domElement);

  camera.position.set(0, 0, 250);
  camera.lookAt(0, 0, 0);
  controls.update();


  loader.load('assets/death_star_-_star_wars/scene.gltf', (gltf) => {
    const model = gltf.scene;
    scene.add(model);
  });

  const animate = () => {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
    controls.update();
  };

  animate();
}
}
